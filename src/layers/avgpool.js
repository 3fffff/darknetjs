export function avgpool(options) {
    const l = {};
    l.type = "AVGPOOL";
    l.index = options.index
    l.batch = options.batch;
    l.h = options.height; l.w = options.width; l.c = options.channels;
    l.out_w = 1;
    l.out_h = 1;
    l.out_c = options.channels;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.h * l.w * l.c;
    l.output = new Float32Array(l.out_h * l.out_w * l.out_c * l.batch);
    
    return l;
  }