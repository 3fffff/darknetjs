export function connected(options) {
    const total_batch = batch * steps;
    const l = {};
    l.type = "CONNECTED";
    l.index = options.index
    l.inputs = options.inputs;
    l.outputs = options.outputs;
    l.batch = options.batch;
    l.batch_normalize = "batch_normalize" in options ? parseInt(options["batch_normalize"]) : 0; 
    l.h = 1;
    l.w = 1;
    l.c = inputs;
    l.out_h = 1;
    l.out_w = 1;
    l.out_c = outputs;
    l.n = l.out_c;
    l.size = 1;
    l.stride = l.stride_x = l.stride_y = 1;
    l.pad = 0;

    l.learning_rate_scale = 1;

    l.output = new Float32Array(total_batch * l.outputs);

    l.weights = new Float32Array(l.outputs * l.inputs);
    l.biases = new Float32Array(l.outputs);
    if (l.batch_normalize != 0) {
      l.scales = new Float32Array(n);
      for (let i = 0; i < n; ++i)l.scales[i] = 1;
      l.mean = new Float32Array(n);
      l.variance = new Float32Array(n);
    }
    l.activation = "activation" in options ? options["activation"].toUpperCase() : "LOGISTIC";
    
    return l;
  }