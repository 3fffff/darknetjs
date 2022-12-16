function avgpool(options, param) {
    const l = {};
    l.type = "AVGPOOL";
    l.batch = options.batch;
    l.h = param.height; l.w = param.width; l.c = param.channels;
    l.out_w = 1;
    l.out_h = 1;
    l.out_c = param.channels;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.h * l.w * l.c;
    l.output = new Float32Array(l.out_h * l.out_w * l.out_c * l.batch);
    
    return l;
  }