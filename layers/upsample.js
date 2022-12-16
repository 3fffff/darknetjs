export function upsample(options, param) {
    let stride = this.option_find_int(options, "stride", 2);
    const l = {};
    l.type = "UPSAMPLE";
    l.tpi = "NEAREST"
    l.batch = options.batch;
    l.w = parseInt(param.width);
    l.h = parseInt(param.height);
    l.c = parseInt(param.channels);
    l.out_w = l.w * stride;
    l.out_h = l.h * stride;
    l.out_c = l.c;
    if (stride < 0) {
      stride = (-1) * stride;
      l.reverse = 1;
      l.out_w = l.w / stride;
      l.out_h = l.h / stride;
    }
    l.stride = stride;
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.inputs = l.w * l.h * l.c;
    l.output = new Float32Array(l.outputs * l.batch);
    l.scale = this.option_find_float(options, "scale", 1);
    if (l.scale == 0) l.scale = 1;
    return l;
  }