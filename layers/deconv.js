export function deconvolutional(options, param) {
  const size = this.option_find_int(options, 'size', 1);
  const filters = this.option_find_int(options, 'filters', 1);
  const pad = this.option_find_int(options, 'pad', 0);
  const padding = pad ? (size >> 1) : "padding" in options ? options["padding"] : 1;
  let stride_x = "stride_x" in options ? options["stride_x"] : -1;
  let stride_y = "stride_x" in options ? options["stride_x"] : -1;
  if (stride_x < 1 || stride_y < 1) {
    const stride = "stride" in options ? options["stride"] : 1;
    stride_x = stride_x < 1 ? stride : stride_x;
    stride_y = stride_y < 1 ? stride : stride_y;
  }
  const l = {};
  l.type = "DECONVOLUTIONAL";
  l.h = parseInt(param.height);
  l.w = parseInt(param.width);
  l.c = parseInt(param.channels);
  l.filters = filters;
  l.batch = options.batch;
  l.stride_x = stride_x;
  l.stride_y = stride_y;
  l.size = size;
  l.pad = padding;
  l.batch_normalize = "batch_normalize" in options ? options["batch_normalize"] : 0;
  l.dilation = "dilation" in options ? options["dilation"] : 1;

  l.weights = new Float32Array(l.c / l.groups * filters * size * size);

  l.biases = new Float32Array(filters);

  l.out_h = Math.floor((l.h + 2 * l.pad - l.size) / l.stride_y) + 1;
  l.out_w = Math.floor((l.w + 2 * l.pad - l.size) / l.stride_x) + 1;
  l.out_c = filters;
  l.outputs = l.out_h * l.out_w * l.out_c;
  l.inputs = l.w * l.h * l.c;

  l.output = new Float32Array(l.batch * l.outputs);
  l.activation = "activation" in options ? options["activation"] : "logistic";

  return l;
}