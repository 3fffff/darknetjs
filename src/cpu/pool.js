export function pool(layers) {
  const l = this
  const input = layers[l.index - 1].output
  for (let b = 0; b < l.batch; ++b) {
    for (let k = 0; k < l.c; ++k) {
      for (let i = 0; i < l.out_h; ++i) {
        for (let j = 0; j < l.out_w; ++j) {
          const out_index = j + l.out_w * (i + l.out_h * (k + l.c * b));
          let avg = 0, counter = 0, valid = false
          if (l.type == 'AVGPOOL') {
            for (let i = 0; i < l.h * l.w; ++i) l.output[out_index] += (input[i + l.h * l.w * (k + b * l.c)]) / (l.h * l.w);
          } else {
            let max = -Number.MAX_VALUE;
            for (let n = 0; n < l.size; ++n) {
              for (let m = 0; m < l.size; ++m) {
                const cur_h = -l.pad + i * l.stride_y + n;
                const cur_w = -l.pad + j * l.stride_x + m;
                const index = cur_w + l.w * (cur_h + l.h * (k + b * l.c));
                valid = (cur_h >= 0 && cur_h < l.h && cur_w >= 0 && cur_w < l.w);
                const val = (valid) ? input[index] : -Number.MAX_VALUE;
                if (l.type == 'LOCALAVG' && valid) {
                  counter++;
                  avg += input[index];
                }
                else max = (val > max) ? val : max;          // get max value
              }
            }
            if (l.type == 'LOCALAVG') l.output[out_index] = (valid) ? avg / counter : l.output[out_index];
            else l.output[out_index] = max;      // store max value
          }
        }
      }
    }
  }
}