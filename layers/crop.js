function crop(options, param) {
    const l = {};
    l.type = "CROP";
    l.batch = param.batch;
    l.h = param.h;
    l.w = param.w;
    l.c = param.c;
    l.scale = param.crop_height / l.h;
    l.flip = param.flip;
    l.angle = param.angle;
    l.saturation = param.saturation;
    l.exposure = param.exposure;
    l.out_w = param.crop_width;
    l.out_h = param.crop_height;
    l.out_c = c;
    l.inputs = l.w * l.h * l.c;
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.output = new Float32Array(l.outputs * batch);
    
    return l;
  }