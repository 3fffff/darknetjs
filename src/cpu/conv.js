import { im2col, matmul, BatchActivate } from './conv_aux.js'

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