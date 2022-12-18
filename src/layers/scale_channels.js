export function scale_channels(options) {
  const lo = "from" in options ? options["from"] : 0;
    let index = parseInt(lo);
    if (index < 0) index = options.index + index;
    else index++
    const l = {};
    l.scale_wh = "scale_wh" in options ? options["scale_wh"] : 0;
    l.type = "SCALE_CHANNELS";
    l.index = options.index
    l.batch = options.batch;
    l.w = options.width;
    l.h = options.height;
    l.c = options.channels;
    l.out_w = options.width;
    l.out_h = options.height;
    l.out_c = options.channels;
    if (l.scale_wh == 0) l.out_c = l.c;
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.inputs = l.outputs;
    l.indexs = index;
    l.output = new Float32Array(l.outputs * l.batch);
    l.activation = "activation" in options ? options["activation"] : "logistic";
    if (l.activation.toUpperCase() == "SWISH" || l.activation.toUpperCase() == "MISH")
      new Error(" [scale_channels] layer doesn't support SWISH or MISH activations ");
    
    return l;
  }