import { batchnorm, gradient } from './conv_aux.js'

export function connected(layers) {
  const l = this
  const input = layers[l.index - 1].output
  const delta = layers[l.index - 1].delta
  gradient(l.output, l.outputs * l.batch, l.activation, l.delta);
  batchnorm(l)
  matmul(l.weights.subarray(w_offset * group), col_buffer_data, input.subarray(y_offset * group), l.filters / l.groups, out_wh, kernel_dim);

  if (delta) matmul(l.weights.subarray(w_offset * group), col_buffer_data, l.delta.subarray(y_offset * group), l.filters / l.groups, out_wh, kernel_dim);
}