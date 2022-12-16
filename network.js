const supportedLayers = ['CONVOLUTIONAL', 'DECONVOLUTIONAL', 'CONNECTED', 'SOFTMAX', 'MAXPOOL', 'AVGPOOL', 'ROUTE', 'YOLO', 'SHORTCUT', 'DROPOUT', 'UPSAMPLE', 'SCALE_CHANNELS', 'SAM', 'CROP', 'COST']

function read_cfg(cfg) {
    const lines = cfg.split("\n");
    const sections = [];
    let section = null;
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].replace(/\s/g, '');
        if (line.length > 0) {
            switch (line[0]) {
                case '#':
                case ';':
                    break;
                case '[': {
                    section = {};
                    section.line = i;
                    section.type = line[line.length - 1] === ']' ? line.substring(1, line.length - 1) : line.substring(1);
                    section.options = {};
                    sections.push(section);
                    break;
                }
                default: {
                    if (!section || line[0] < 0x20 || line[0] > 0x7E) throw new Error("Invalid cfg '" + text.replace(/[^\x20-\x7E]+/g, '').trim() + "' at line " + i.toString() + ".");
                    if (section) {
                        const index = line.indexOf('=');
                        if (index < 0) throw new Error("Invalid cfg '" + text.replace(/[^\x20-\x7E]+/g, '').trim() + "' at line " + i.toString() + ".");
                        const key = line.substring(0, index);
                        const value = line.substring(index + 1);
                        section.options[key] = value;
                    }
                    break;
                }
            }
        }
    }
    return sections;
}
function load_weights_upto_cpu(layers, data, merge, quant) {
    let offset = 0, counter = 0
    const major = data.getInt32(offset, true);
    offset += 4
    const minor = data.getInt32(offset, true);
    offset += 8
    offset = ((major * 10 + minor) >= 2) ? offset + 8 : offset + 4
    for (let j = 1; j < layers.length; ++j) {
        if (layers[j].dontload != 0) continue;
        if (layers[j].type == "CONVOLUTIONAL" || layers[j].type == "DECONVOLUTIONAL" || layers[j].type == "CONNECTED") {
            const l = layers[j]
            const num = l.filters * l.c / l.groups * l.size * l.size;
            for (let i = 0; i < l.filters; i++) { l.biases[i] = data.getFloat32(offset, true); offset += 4 }
            if (l.batch_normalize == 1) {
                for (let i = 0; i < l.filters; i++) { l.scales[i] = data.getFloat32(offset, true); offset += 4 }
                for (let i = 0; i < l.filters; i++) { l.mean[i] = data.getFloat32(offset, true); offset += 4 }
                for (let i = 0; i < l.filters; i++) { l.variance[i] = data.getFloat32(offset, true); offset += 4 }
            }
            for (let i = 0; i < num; i++) { l.weights[i] = data.getFloat32(offset, true); offset += 4; }
            if (l.batch_normalize == 1 && merge == 1) {
                for (let f = 0; f < l.filters; ++f) {
                    l.biases[f] = l.biases[f] - l.scales[f] * l.mean[f] / (Math.sqrt(l.variance[f] + 0.00001));
                    const precomputed = l.scales[f] / (Math.sqrt(l.variance[f] + 0.00001));
                    const filter_size = l.size * l.size * l.c / l.groups;
                    for (let i = 0; i < filter_size; ++i) l.weights[f * filter_size + i] *= precomputed;
                }
                l.scales = null; l.mean = null; l.variance = null;
                l.batch_normalize = 0;
            }

            if (l.type === "CONVOLUTIONAL" && l.batch_normalize === 0 && quant) {
                l.input_quant_multipler = (counter < layers[0].input_calibration.length) ? layers[0].input_calibration[counter] : 40;
                counter++;
                l.weights_quant_multipler = get_multiplier(l.weights, 8) / 4;    // good [2 - 8], best 4
                l.weights_quant = new Int8Array(l.weights.length)
                l.biases_quant = new Int8Array(l.biases.length)
                const num = l.c / l.groups * l.size * l.size;
                for (let fil = 0; fil < num; ++fil) {
                    for (let i = 0; i < l.filters; ++i) {
                        const w = l.weights[fil * l.filters + i] * l.weights_quant_multipler;
                        l.weights_quant[fil * l.filters + i] = Math.max(Math.abs(w), 127);
                    }
                }
                const R_MULT = 32
                l.output_multipler = l.input_quant_multipler / (l.weights_quant_multipler * l.input_quant_multipler / R_MULT);
                l.ALPHA1 = 1 / (l.input_quant_multipler * l.weights_quant_multipler);
                for (let fil = 0; fil < num; ++fil) {
                    const biases_multipler = l.output_multipler * l.weights_quant_multipler * l.input_quant_multipler / R_MULT;
                    l.biases_quant[fil] = l.biases[fil] * biases_multipler;
                }
            }
        }
    }
}

function get_multiplier(arr, bits_length) {
    const number_of_ranges = 32;
    const start_range = 1 / 65536;
    //distribution
    const count = new Int16Array(number_of_ranges);
    for (let i = 0; i < arr.length; ++i) {
        let cur_range = start_range;
        for (let j = 0; j < number_of_ranges; ++j) {
            if (Math.abs(cur_range) <= arr[i] && arr[i] < Math.abs(cur_range * 2))
                count[j]++;
            cur_range *= 2;
        }
    }

    let max_count_range = 0;
    let index_max_count = 0;
    for (let j = 0; j < number_of_ranges; ++j) {
        let counter = 0;
        for (let i = j; i < (j + bits_length) && i < number_of_ranges; ++i)
            counter += count[i];
        if (max_count_range < counter) {
            max_count_range = counter;
            index_max_count = j;
        }
    }
    let multiplier = 1 / (start_range * Math.pow(2.0, index_max_count));
    return multiplier;
}

export function parse_network_cfg(filename, weights, quant, wasm, webgl) {
    let sections = read_cfg(filename);
    const layers = [];
    if (sections[0].type == "[net]" && sections[0].type == "[network]") throw new Error("First section must be [net] or [network]");
    let net = {}
    net.w = "width" in sections[0].options ? sections[0].options["width"] : 1;
    net.h = "height" in sections[0].options ? sections[0].options["height"] : 1;
    net.c = "channels" in sections[0].options ? sections[0].options["channels"] : 1;
    net.batch = "batch" in sections[0].options ? sections[0].options["batch"] : 1;
    net.train = "train" in sections[0].options ? sections[0].options["train"] : 1;
    net.quant = quant
    net.wasmSupport = wasm
    if (quant) {
        const input_calibration = sections[0].options.input_calibration.split(',')
        net.input_calibration = new Float32Array(input_calibration.length)
        for (let i = 0; i < input_calibration.length; i++)
            net.input_calibration[i] = parseFloat(input_calibration[i])
    }
    layers.push(net)
    for (let i = 1; i < sections.length; i++) {
        let l = {};
        sections[i].options.index = i
        sections[i].options.batch = net.batch
        l.dontload = "dontload" in options ? options["dontload"] : 0; 
        l.dontloadscales = "dontloadscales" in options ? options["dontloadscales"] : 0; 
        l.index = i
        if (!l.type) continue
        layers.push(l);
    }
    load_weights_upto_cpu(layers, weights, 1, net.quant)
    console.log(layers)
    return layers
}