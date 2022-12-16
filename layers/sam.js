export function sam(options, layers) {
  const lo = "from" in options ? options["from"] : 0;
    let index = parseInt(lo);
    if (index < 0) index = options.index + index;
    else index++
    let from = layers[index];

    const l = {}
    l.type = "SAM";
    l.batch = options.batch;
    l.w = layers[options.index - 1].out_w;
    l.h = layers[options.index - 1].out_h;
    l.c = layers[options.index - 1].out_c;

    l.out_w = from.out_w;
    l.out_h = from.out_h;
    l.out_c = from.out_c;

    l.outputs = l.out_w * l.out_h * l.out_c;
    l.inputs = l.outputs;
    l.indexs = from.index;
    l.input_layers = templayers;

    l.output = new Float32Array(l.outputs * options.batch);
    l.activation = "activation" in options ? options["activation"] : "logistic";
    if (l.activation == 'SWISH' || l.activation == 'MISH')
      console.log(" [sam] layer doesn't support SWISH or MISH activations \n");
    
    return l;
  }