export function sam(options) {
  const lo = "from" in options ? options["from"] : 0;
    let index = parseInt(lo);
    if (index < 0) index = options.index + index;
    else index++

    const l = {}
    l.type = "SAM";
    l.index = options.index
    l.batch = options.batch;
    l.w = options.width;
    l.h = options.height;
    l.c = options.channels;
    l.out_w = options.width;
    l.out_h = options.height;
    l.out_c = options.channels;
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.inputs = l.outputs;
    l.indexs = index;

    l.output = new Float32Array(l.outputs * options.batch);
    l.activation = "activation" in options ? options["activation"] : "logistic";
    if (l.activation == 'SWISH' || l.activation == 'MISH')
      console.log(" [sam] layer doesn't support SWISH or MISH activations \n");
    
    return l;
  }