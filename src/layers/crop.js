export function crop(options) {
    const l = {};
    l.index = options.index
    l.type = "CROP";
    l.batch = options.batch;
    l.h = options.h;
    l.w = options.w;
    l.c = options.c;
    l.scale = options.crop_height / l.h;
    l.flip = options.flip;
    l.angle = options.angle;
    l.saturation = options.saturation;
    l.exposure = options.exposure;
    l.out_w = options.crop_width;
    l.out_h = options.crop_height;
    l.out_c = c;
    l.inputs = l.w * l.h * l.c;
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.output = new Float32Array(l.outputs * batch);
    
    return l;
  }