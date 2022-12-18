export function dropout_layer(layers) {
    const l = this
    if (!delta) return;
    for (let i = 0; i < l.batch * l.inputs; ++i) {
      const r = l.rand[i];
      if (r < l.probability) delta[i] = 0;
      else layers[l.index - 1].delta[i] *= l.scale;
    }
  }