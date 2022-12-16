export function softmax_layer(layers) {
    const l = this
    for (let i = 0; i < l.inputs * l.batch; ++i) layers[l.index - 1].delta[i] += l.delta[i];
  }
