export function dropout(options, param) {
    const probability = "probability" in options ? options["probability"] : 0; 
    const l = {};
    l.type = "DROPOUT";
    l.probability = probability;
    l.inputs = param.inputs;
    l.outputs = param.inputs;
    l.batch = options.batch;
    l.scale = (1.0 / (1.0 - probability));
    l.out_w = parseInt(param.width);
    l.out_h = parseInt(param.height);
    l.out_c = parseInt(param.channels);
    
    return l;
}