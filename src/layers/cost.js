export function cost(options) {
    const l = {};
    l.type = "COST";
    l.index = options.index
    l.scale = options.scale;
    l.batch = options.batch;
    l.inputs = options.inputs;
    l.outputs = options.inputs;
    l.cost_type = options.cost_type;
    l.delta = new Float32Array(inputs * batch);
    l.output = new Float32Array(inputs * batch);
    l.cost = 0;
    
    return l;
  }