export function shortcut_layer(layers) {
    const l = this
    Backward.gradient(l.output, l.outputs * l.batch, l.activation, l.delta);
    for (let i = 0; i < l.outputs * l.batch; ++i) layers[i - 1].delta[i] += l.delta[i];
    shortcut_cpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta, l.w, l.h, l.c, 1, l.beta, layers[l.index].delta);

  }