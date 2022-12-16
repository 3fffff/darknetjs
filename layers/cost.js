function cost(options, param) {
    const l = {};
    l.type = "COST";

    l.scale = param.scale;
    l.batch = param.batch;
    l.inputs = param.inputs;
    l.outputs = param.inputs;
    l.cost_type = param.cost_type;
    l.delta = new Float32Array(inputs * batch);
    l.output = new Float32Array(inputs * batch);
    l.cost = 0;
    
    return l;
  }