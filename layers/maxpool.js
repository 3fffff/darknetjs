export function maxpool(options, param) {
  const stride = this.option_find_int(options, "stride", 1);
  const size = this.option_find_int(options, "size", stride);
  const padding = Math.floor(this.option_find_int(options, "padding", (size - 1)));
  const antialiasing = this.option_find_int(options, 'antialiasing', 0);
  const blur_stride_x = this.option_find_int(options, 'stride_x', stride);
  const blur_stride_y = this.option_find_int(options, 'stride_y', stride);
  const stride_x = antialiasing ? 1 : blur_stride_x;
  const stride_y = antialiasing ? 1 : blur_stride_y;

  const l = {};
  l.type = "MAXPOOL";
  l.batch = options.batch;
  l.h = param.height; l.w = param.width; l.c = param.channels;
  l.pad = padding;
  l.out_w = Math.floor((l.w + padding - size) / stride_x) + 1;
  l.out_h = Math.floor((l.h + padding - size) / stride_y) + 1;
  l.out_c = l.c;
  l.outputs = l.out_h * l.out_w * l.out_c;
  l.inputs = l.h * l.w * l.c;
  l.size = size;
  l.stride_x = stride_x;
  l.stride_y = stride_y
  l.output = new Float32Array(l.out_h * l.out_w * l.out_c * l.batch);
  
  return l;
}