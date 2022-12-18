export function cost(layers) {
  const l = this
  const cost_type = []
  if (!l.truth) return;
  if (l.cost_type == MASKED) {
    for (let i = 0; i < l.batch * l.inputs; ++i) {
      if (l.truth[i] == SECRET_NUM) layers[l.index-1][i] = SECRET_NUM;
    }
  }
  if (l.cost_type == SMOOTH) {
    smooth_l1_cpu(l.batch * l.inputs, layers[l.index-1], layers[l.index-1].truth, l.delta, l.output);
  } else {
    l2_cpu(l.batch * l.inputs, layers[l.index-1], layers[l.index-1].truth, l.delta, l.output);
  }
  l.cost = l.output.reduce((psum, el) => psum + el, 0);
}