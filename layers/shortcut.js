export function shortcut(options, layers) {
    const lo = "from" in options ? options["from"] : 0;
    let index = parseInt(lo);
    if (index < 0) index = options.index + index;
    else index++

    const from = layers[index];
    const l = {};
    l.type = "SHORTCUT";
    l.batch = options.batch;
    l.w = from.out_w;
    l.h = from.out_h;
    l.c = from.out_c;
    l.out_w = layers[options.index - 1].out_w;
    l.out_h = layers[options.index - 1].out_h;
    l.out_c = layers[options.index - 1].out_c;
    l.outputs = l.w * l.h * l.c;
    l.inputs = l.outputs;
    l.indexs = from.index;
    l.output = new Float32Array(l.outputs * l.batch);
    
    l.activation = options.activation.toUpperCase();
    return l;
  }