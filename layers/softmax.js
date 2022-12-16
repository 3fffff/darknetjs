function softmax(options, param) {
    const l = {};
    l.type = 'SOFTMAX';
    l.batch = options.batch;
    l.inputs = param.width * param.height * param.channels;
    l.outputs = l.inputs;
    l.groups = "groups" in options ? options["groups"] : 0; 
    l.output = new Float32Array(l.inputs * l.batch);
    
    return l;
}