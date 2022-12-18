export function crop(layers) {
  count = 0;
  let flip = (l.flip && Math.random() % 2);
  dh = Math.random() % (l.h - l.out_h + 1);
  dw = Math.random() % (l.w - l.out_w + 1);
  scale = 2;
  trans = -1;
  if (l.noadjust) {
    scale = 1;
    trans = 0;
  }
  if (!layers[0].train) {
    flip = 0;
    dh = (l.h - l.out_h) / 2;
    dw = (l.w - l.out_w) / 2;
  }
  for (let b = 0; b < l.batch; ++b) {
    for (let c = 0; c < l.c; ++c) {
      for (let i = 0; i < l.out_h; ++i) {
        for (let j = 0; j < l.out_w; ++j) {
          if (flip)
            col = l.w - dw - j - 1;
          else
            col = j + dw;
          row = i + dh;
          const index = col + l.w * (row + l.h * (c + l.c * b));
          l.output[count++] = state.input[index] * scale + trans;
        }
      }
    }
  }
}