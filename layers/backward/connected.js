export function connected_layer(layers) {
    const l = this
    Backward.gradient(l.output, l.outputs * l.batch, l.activation, l.delta);
    Backward.batchnorm(l)

    matmul(l.weights.subarray(w_offset * group), col_buffer_data, l.output.subarray(y_offset * group), l.filters / l.groups, out_wh, kernel_dim);

    if (layers[l.index - 1].delta) matmul(l.weights.subarray(w_offset * group), col_buffer_data, l.delta.subarray(y_offset * group), l.filters / l.groups, out_wh, kernel_dim);
  }