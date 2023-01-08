export function dropout(options) {
    const probability = "probability" in options ? options["probability"] : 0;
    const l = {};
    l.type = "DROPOUT";
    l.index = options.index
    l.probability = probability;

    l.batch = options.batch;
    l.scale = (1.0 / (1.0 - probability));
    l.out_w = parseInt(options.width);
    l.out_h = parseInt(options.height);
    l.out_c = parseInt(options.channels);
    l.inputs = l.out_w * l.out_h * l.out_c * l.batch;
    l.outputs = l.inputs;
    return l;
}