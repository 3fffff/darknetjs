export function convolutional(options) {
  const size = "size" in options ? parseInt(options["size"]) : 1; 
  const filters = "filters" in options ? parseInt(options["filters"]) : 1; 
  const pad = "pad" in options ? parseInt(options["pad"]) : 0; 
  const padding = pad ? (size >> 1) : "padding" in options ? parseInt(options["padding"]) : 0; 
  let stride_x = "stride_x" in options ? parseInt(options["stride_x"]) : -1; 
  let stride_y = "stride_x" in options ? parseInt(options["stride_x"]) : -1; 
  if (stride_x < 1 || stride_y < 1) {
    const stride = "stride" in options ? parseInt(options["stride"]) : 1; 
    stride_x = stride_x < 1 ? stride : stride_x;
    stride_y = stride_y < 1 ? stride : stride_y;
  }
  const l = {};
  l.type = "CONVOLUTIONAL";
  l.index = options.index
  l.h = parseInt(options.height);
  l.w = parseInt(options.width);
  l.c = parseInt(options.channels);
  l.filters = filters;
  l.groups = "groups" in options ? parseInt(options["groups"]) : 1; 
  l.batch = options.batch;
  l.stride_x = stride_x;
  l.stride_y = stride_y;
  l.size = size;
  l.pad = padding;
  l.batch_normalize = "batch_normalize" in options ? parseInt(options["batch_normalize"]): 0; 
  l.dilation = "dilation" in options ? options["dilation"] : 1; 

  l.weights = new Float32Array(l.c / l.groups * filters * size * size);

  l.biases = new Float32Array(filters);

  l.out_h = Math.floor((l.h + 2 * l.pad - l.size) / l.stride_y) + 1;
  l.out_w = Math.floor((l.w + 2 * l.pad - l.size) / l.stride_x) + 1;
  l.out_c = filters;
  l.outputs = l.out_h * l.out_w * l.out_c;
  l.inputs = l.w * l.h * l.c;

  l.output = new Float32Array(l.batch * l.outputs);

  if (l.batch_normalize) {
    l.scales = new Float32Array(filters);
    for (let i = 0; i < filters; ++i)l.scales[i] = 1;
    l.mean = new Float32Array(filters);
    l.variance = new Float32Array(filters);
  }
  l.activation = "activation" in options ? options["activation"].toUpperCase() : "LOGISTIC";

  return l;
}