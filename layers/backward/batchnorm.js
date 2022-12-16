export function batchnorm(l) {
    //backward bias
    for (let b = 0; b < batch; ++b) {
      for (let i = 0; i < n; ++i) {
        l.bias_updates[i] += sum_array(delta + size * (i + b * n), size);
        //scale bias
        if (l.batch_norm) {
          for (let j = 0; j < size; ++j) {
            l.delta[(b * n + i) * size + j] *= l.scales[i];
          }
        }
      }
    }
    if (l.batch_norm) {
      for (let f = 0; f < l.filters; ++f) {
        let sum = 0;
        l.variance_delta[f] = 0;
        l.mean_delta[f] = 0;
        for (let j = 0; j < l.batch; ++j) {
          for (let s = 0; s < size; ++s) {
            const index = s + size * (f + n * b);
            sum += l.delta[index] * l.x_norm[index];
          }
          for (let k = 0; k < l.spatial; ++k) {
            const index = j * l.filters * l.spatial + f * l.spatial + k;
            l.variance_delta[f] += l.delta[index] * (l.x[index] - l.mean[f]);
            l.mean_delta[f] += l.delta[index];
          }
        }
        l.variance_delta[f] *= -0.5 * Math.pow(l.variance[i] + 0.00001, (-3 / 2));
        l.mean_delta[f] *= (-1 / Math.sqrt(l.variance[i] + 0.00001));
        l.scale_updates[f] += sum;
      }

      //normalize_delta
      for (let j = 0; j < batch; ++j) {
        for (let f = 0; f < filters; ++f) {
          for (let k = 0; k < spatial; ++k) {
            const index = j * filters * spatial + f * spatial + k;
            l.delta[index] = l.delta[index] * 1 / (Math.sqrt(l.variance[f] + 0.00001)) + l.variance_delta[f] * 2 * (l.x[index] - l.mean[f]) / (l.spatial * l.batch) + l.mean_delta[f] / (l.spatial * l.batch);
          }
        }
      }
    }
  }