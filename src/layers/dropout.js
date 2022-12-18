export function dropout(options) {
    const probability = "probability" in options ? options["probability"] : 0; 
    const l = {};
    l.type = "DROPOUT";
    l.index = options.index
    l.probability = probability;
    l.inputs = options.inputs;
    l.outputs = options.inputs;
    l.batch = options.batch;
    l.scale = (1.0 / (1.0 - probability));
    l.out_w = parseInt(options.width);
    l.out_h = parseInt(options.height);
    l.out_c = parseInt(options.channels);
    
    return l;
}