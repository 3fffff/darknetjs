
export function convolutional(layers) {
  const l = this
  const input = layers[l.index - 1].output
  const kernel_dim = l.size * l.size * l.c / l.groups;
  const out_wh = l.out_h * l.out_w;

  const col_buffer_data = new Float32Array(kernel_dim * l.out_h * l.out_w);
  const x_offset = l.c * l.h * l.w / l.groups;
  const y_offset = l.out_w * l.out_h * l.out_c / l.groups;
  const w_offset = l.filters * kernel_dim / l.groups;
  const out_size = l.out_h * l.out_w;
  for (let b = 0; b < l.batch; ++b) {
    for (let group = 0; group < l.groups; ++group) {
      im2col(input.subarray(x_offset * group), col_buffer_data, l.c / l.groups, l.h, l.w, l.size, l.size, l.dilation, l.dilation, l.pad, l.pad, l.pad, l.pad, l.stride_y, l.stride_x);
      matmul(l.weights.subarray(w_offset * group), col_buffer_data, l.output.subarray(y_offset * group), l.filters / l.groups, out_wh, kernel_dim);
    }
  }
  BatchActivate(l, out_size)
}

export function convolutional_layer_q(layers) {
  const l = this
  const input = l.index - 1 == 0 ? layers[l.index - 1].output : new Int8Array(layers[l.index - 1].output.length)
  if (l.index - 1 != 0) {
    for (let i = 0; i < layers[l.index - 1].output.length; ++i) {
      const src = layers[l.index - 1].output[i] * l.input_quant_multipler;
      input[i] = Math.max(Math.abs(src), 127);
    }
  }
  const kernel_dim = l.size * l.size * l.c / l.groups;
  const out_wh = l.out_h * l.out_w;

  const col_buffer_data = new Int8Array(kernel_dim * l.out_h * l.out_w);
  const x_offset = l.c * l.h * l.w / l.groups;
  const y_offset = l.out_w * l.out_h * l.out_c / l.groups;
  const w_offset = l.filters * kernel_dim / l.groups;
  const output_q = new Int32Array(l.output.length)
  for (let b = 0; b < l.batch; ++b) {
    for (let group = 0; group < l.groups; ++group) {
      im2col(input.subarray(x_offset * group), col_buffer_data, l.c / l.groups, l.h, l.w, l.size, l.size, l.dilation, l.dilation, l.pad, l.pad, l.pad, l.pad, l.stride_y, l.stride_x);
      matmul(l.weights_quant.subarray(w_offset * group), col_buffer_data, output_q.subarray(y_offset * group), l.filters / l.groups, out_wh, kernel_dim);
    }
  }
  for (let i = 0; i < l.outputs; ++i) l.output[i] = output_q[i] * l.ALPHA1;
  BatchActivate(l, out_size)
}
