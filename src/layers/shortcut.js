export function shortcut(options) {
    const lo = "from" in options ? options["from"] : 0;
    let index = parseInt(lo);
    if (index < 0) index = options.index + index;
    else index++

    const l = {};
    l.type = "SHORTCUT";
    l.index = options.index
    l.batch = options.batch;
    l.w = options.width;
    l.h = options.height;
    l.c = options.channels;
    l.out_w = options.width;
    l.out_h = options.height;
    l.out_c = options.channels;
    l.outputs = l.w * l.h * l.c;
    l.inputs = l.outputs;
    l.indexs = index;
    l.output = new Float32Array(l.outputs * l.batch);
    
    l.activation = options.activation.toUpperCase();
    return l;
  }