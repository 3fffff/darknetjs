export function connected(layers) {
  const l = this
  const input = layers[l.index - 1].output
  matmul(input, l.weights, l.output, l.batch, l.inputs, l.outputs);
  BatchActivate(l, l.outputs)
}