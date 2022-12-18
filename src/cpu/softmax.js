export function softmax(layers) {
  const l = this
  const input = layers[l.index - 1].output
  for (let b = 0; b < l.batch; ++b) {
    for (let g = 0; g < l.groups; ++g) {
      let sum = 0;
      let largest = -Number.MAX_VALUE;
      for (let i = index; i < index + n; ++i)if (input[i] > largest) largest = input[i];
      for (let i = index; i < index + n; ++i) {
        const e = Math.exp(input[i] / temp - largest / temp);
        sum += e;
        output[i] = e;
      }
      for (let i = index; i < index + n; ++i)output[i] /= sum;
    }
  }
}