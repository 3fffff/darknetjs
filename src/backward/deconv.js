import { batchnorm, gradient } from './conv_aux.js'

export function deconvolutional(layers) {
  const l = this
  const input = layers[l.index - 1].output
  const delta = layers[l.index - 1].delta
  const kernel_dim = l.size * l.size * l.c / l.groups;
  const out_wh = l.out_h * l.out_w;

  const col_buffer_data = new Float32Array(kernel_dim * l.out_h * l.out_w);
  const x_offset = l.c / l.groups * l.h * l.w;
  const y_offset = l.out_w * l.out_h * l.out_c / l.groups;
  const w_offset = l.filters * kernel_dim / l.groups;

  gradient(l.output, l.outputs * l.batch, l.activation, l.delta);
  batchnorm(l)

  for (let b = 0; b < l.batch; ++b) {
    for (let group = 0; group < l.groups; ++group) {
      matmul(l.weights.subarray(w_offset * group), input.subarray(x_offset * group), col_buffer_data, l.filters / l.groups, out_wh, kernel_dim);
      im2col(col_buffer_data, l.output.subarray(y_offset * group), l.out_c / l.groups, l.out_h, l.out_w, l.size, l.size, l.dilation, l.dilation, l.pad, l.pad, l.pad, l.pad, l.stride_y, l.stride_x);
      if (delta) {
        matmul(l.weights.subarray(w_offset * group), delta.subarray(x_offset * group), col_buffer_data, l.filters / l.groups, out_wh, kernel_dim);
        im2col(col_buffer_data, l.delta.subarray(y_offset * group), l.out_c / l.groups, l.out_h, l.out_w, l.size, l.size, l.dilation, l.dilation, l.pad, l.pad, l.pad, l.pad, l.stride_y, l.stride_x);
      }
    }
  }
}