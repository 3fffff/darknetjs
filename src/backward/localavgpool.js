export function local_avgpool(layers) {
  const l = this
  const w_offset = -l.pad / 2;
  const h_offset = -l.pad / 2;
  for (let b = 0; b < l.batch; ++b) {
    for (let k = 0; k < l.c; ++k) {
      for (let i = 0; i < l.h; ++i) {
        for (let j = 0; j < l.w; ++j) {
          const out_index = j + w * (i + h * (k + c * b));
          for (let n = 0; n < l.size; ++n) {
            for (let m = 0; m < l.size; ++m) {
              const cur_h = h_offset + i * l.stride_y + n;
              const cur_w = w_offset + j * l.stride_x + m;
              const index = cur_w + l.w * (cur_h + l.h * (k + b * l.c));
              const valid = (cur_h >= 0 && cur_h < l.h &&
                cur_w >= 0 && cur_w < l.w);
              if (valid) layers[l.index - 1].delta[index] += l.delta[out_index] / (l.size * l.size);
            }
          }
        }
      }
    }
  }
}