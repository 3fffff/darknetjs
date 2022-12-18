export function softmax(options) {
    const l = {};
    l.type = 'SOFTMAX';
    l.index = options.index
    l.batch = options.batch;
    l.inputs = param.width * param.height * param.channels;
    l.outputs = l.inputs;
    l.groups = "groups" in options ? parseInt(options["groups"]) : 0; 
    l.output = new Float32Array(l.inputs * l.batch);
    
    return l;
}