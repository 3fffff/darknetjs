export function scale_channels(options, layers) {
  const lo = "from" in options ? options["from"] : 0;
    let index = parseInt(lo);
    if (index < 0) index = options.index + index;
    else index++
    const l = {};
    l.scale_wh = "scale_wh" in options ? options["scale_wh"] : 0;
    const from = layers[index];
    l.type = "SCALE_CHANNELS";
    l.batch = options.batch;
    l.w = layers[options.index - 1].out_w;
    l.h = layers[options.index - 1].out_h;
    l.c = layers[options.index - 1].out_c;
    l.out_w = from.out_w;
    l.out_h = from.out_h;
    l.out_c = from.out_c;
    if (l.scale_wh == 0) l.out_c = l.c;
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.inputs = l.outputs;
    l.indexs = from.index;
    l.output = new Float32Array(l.outputs * l.batch);
    l.activation = "activation" in options ? options["activation"] : "logistic";
    if (l.activation.toUpperCase() == "SWISH" || l.activation.toUpperCase() == "MISH")
      new Error(" [scale_channels] layer doesn't support SWISH or MISH activations ");
    
    return l;
  }