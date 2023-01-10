export function sam(layers) {
  const l = this
  const from_output = layers[l.indexs].output;
  const from_delta = layers[l.indexs].delta;
  for (i = 0; i < l.batch * l.out_c * l.out_w * l.out_h; ++i) {
    layers[l.index - 1].delta[i] += l.delta[i] * from_output[i];
    from_delta[i] = layers[l.index - 1].output[i] * l.delta[i]; // input * l.delta
  }
}