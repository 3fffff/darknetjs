export function maxpool(layers) {
    const l = this
    for (let i = 0; i < l.out_h * l.out_w * l.c * l.batch; ++i) {
      const index = l.indexes[i];
      layers[l.index - 1].delta[index] += l.delta[i];
    }
  }