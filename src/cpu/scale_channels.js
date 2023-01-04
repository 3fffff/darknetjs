export function scale_channels(layers) {
  for (let i = 0; i < this.output.length; ++i)this.output[i] = layers[this.index - 1].output[~~(i / (this.out_w * this.out_h))] * layers[this.indexs].output[i];
  this.output = activate(this.output, this.activation);
}