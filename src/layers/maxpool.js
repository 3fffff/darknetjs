export function maxpool(options) {
  const stride = "stride" in options ? parseInt(options["stride"]) : 1;
  const size = "size" in options ? parseInt(options["size"]) : stride;
  const padding = Math.floor("padding" in options ? parseInt(options["padding"]) : (size - 1));
  const antialiasing = "antialiasing" in options ? parseInt(options["antialiasing"]) : 0;
  const blur_stride_x = "stride_x" in options ? parseInt(options["stride_x"]) : stride;
  const blur_stride_y = "stride_x" in options ? parseInt(options["stride_x"]) : stride;
  const stride_x = antialiasing ? 1 : blur_stride_x;
  const stride_y = antialiasing ? 1 : blur_stride_y;

  const l = {};
  l.type = "MAXPOOL";
  l.index = options.index
  l.batch = options.batch;
  l.h = options.height; l.w = options.width; l.c = options.channels;
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