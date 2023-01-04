export function scale_channels(layers) {
    const l = this
    Backward.gradient(l.output, l.outputs * l.batch, l.activation, l.delta);

    const size = l.batch * l.out_c * l.out_w * l.out_h;
    const channel_size = l.out_w * l.out_h;
    const from_output = layers[l.index - 1].output;
    const from_delta = layers[l.index - 1].delta;

    for (let i = 0; i < size; ++i) {
      layers[l.index].delta[i / channel_size] += l.delta[i] * from_output[i];// / channel_size; // l.delta * from  (should be divided by channel_size?)
      from_delta[i] += l.output[i / channel_size] * l.delta[i]; // input * l.delta
    }
  }