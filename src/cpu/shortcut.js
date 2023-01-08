export function shortcut(layers) {
  for (let i = 0; i < this.output.length; ++i)this.output[i] = layers[this.index - 1].output[i] + layers[this.indexs].output[i];
}