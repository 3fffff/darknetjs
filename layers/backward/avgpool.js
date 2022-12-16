export function avgpool_layer(layers) {
    const l = this
    for (let b = 0; b < l.batch; ++b) {
      for (let k = 0; k < l.c; ++k) {
        const out_index = k + b * l.c;
        for (let i = 0; i < l.h * l.w; ++i) {
          const in_index = i + l.h * l.w * (k + b * l.c);
          layers[l.index - 1].delta[in_index] += l.delta[out_index] / (l.h * l.w);
        }
      }
    }
  }