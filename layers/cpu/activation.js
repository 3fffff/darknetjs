function im2col(data_im, data_col, channels, height,
    width, kernel_h, kernel_w, dilation_h, dilation_w, pad_t,
    pad_l, pad_b, pad_r, stride_h, stride_w) {
  
    const dkernel_h = dilation_h * (kernel_h - 1) + 1;
    const dkernel_w = dilation_w * (kernel_w - 1) + 1;
  
    const height_col = ~~((height + pad_t + pad_b - dkernel_h) / stride_h) + 1;
    const width_col = ~~((width + pad_l + pad_r - dkernel_w) / stride_w) + 1;
  
    const channels_col = channels * kernel_h * kernel_w;
    for (let c = 0; c < channels_col; ++c) {
      const w_offset = c % kernel_w;
      const h_offset = ~~(c / kernel_w) % kernel_h;
      const c_im = ~~(c / (kernel_h * kernel_w));
      for (let h = 0; h < height_col; ++h) {
        for (let w = 0; w < width_col; ++w) {
          const h_pad = h * stride_h - pad_t + h_offset * dilation_h;
          const w_pad = w * stride_w - pad_l + w_offset * dilation_w;
          if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
            data_col[(c * height_col + h) * width_col + w] = data_im[(c_im * height + h_pad) * width + w_pad];
          else data_col[((c * height_col + h) * width_col + w)] = 0;
        }
      }
    }
  }
  function matmul(A, B, C, M, N, K) {
    const alignedN = ~~(N - N % 4);
    const alignedM = ~~(M - M % 4);
    for (let m = 0; m < alignedM; m += 4) {
      for (let n = 0; n < alignedN; n += 4) {
        let C00 = 0, C01 = 0, C02 = 0, C03 = 0, C10 = 0, C11 = 0, C12 = 0, C13 = 0, C20 = 0, C21 = 0, C22 = 0, C23 = 0, C30 = 0, C31 = 0, C32 = 0, C33 = 0;
        for (let k = 0; k < K; k += 1) {
          C00 += A[(m + 0) * K + k] * B[k * N + n + 0];
          C01 += A[(m + 0) * K + k] * B[k * N + n + 1];
          C02 += A[(m + 0) * K + k] * B[k * N + n + 2];
          C03 += A[(m + 0) * K + k] * B[k * N + n + 3];
  
          C10 += A[(m + 1) * K + k] * B[k * N + n + 0];
          C11 += A[(m + 1) * K + k] * B[k * N + n + 1];
          C12 += A[(m + 1) * K + k] * B[k * N + n + 2];
          C13 += A[(m + 1) * K + k] * B[k * N + n + 3];
  
          C20 += A[(m + 2) * K + k] * B[k * N + n + 0];
          C21 += A[(m + 2) * K + k] * B[k * N + n + 1];
          C22 += A[(m + 2) * K + k] * B[k * N + n + 2];
          C23 += A[(m + 2) * K + k] * B[k * N + n + 3];
  
          C30 += A[(m + 3) * K + k] * B[k * N + n + 0];
          C31 += A[(m + 3) * K + k] * B[k * N + n + 1];
          C32 += A[(m + 3) * K + k] * B[k * N + n + 2];
          C33 += A[(m + 3) * K + k] * B[k * N + n + 3];
        }
        C[(m + 0) * N + n + 0] = C00, C[(m + 0) * N + n + 1] = C01, C[(m + 0) * N + n + 2] = C02, C[(m + 0) * N + n + 3] = C03;
        C[(m + 1) * N + n + 0] = C10, C[(m + 1) * N + n + 1] = C11, C[(m + 1) * N + n + 2] = C12, C[(m + 1) * N + n + 3] = C13;
        C[(m + 2) * N + n + 0] = C20, C[(m + 2) * N + n + 1] = C21, C[(m + 2) * N + n + 2] = C22, C[(m + 2) * N + n + 3] = C23;
        C[(m + 3) * N + n + 0] = C30, C[(m + 3) * N + n + 1] = C31, C[(m + 3) * N + n + 2] = C32, C[(m + 3) * N + n + 3] = C33;
      }
      for (let j = alignedN; j < N; ++j)
        for (let k = 0; k < K; ++k)
          C[m * N + j] += A[m * K + k] * B[k * N + j];
    }
    for (let i = alignedM; i < M; ++i)
      for (let k = 0; k < K; ++k)
        for (let j = 0; j < N; ++j)
          C[i * N + j] += A[i * K + k] * B[k * N + j];
  }
  function BatchActivate(l, out_size) {
    for (let nc = 0; nc < l.batch * l.filters; nc++) {
      const offset = nc * out_size;
      if (l.batch_normalize == 1)
        for (let i = 0; i < out_size; i++)l.output[offset + i] = l.scales[nc % l.filters] * ((l.output[offset + i] - l.mean[nc % l.filters]) / Math.sqrt(l.variance[nc % l.filters] + 0.000001)) + l.biases[nc % l.filters];
      else if (l.biases.length != 0) for (let i = 0; i < out_size; i++)l.output[offset + i] += l.biases[nc % l.filters];
    }
    l.output = activate(l.output, l.activation);
  }

function activate(x, a) {
  switch (a) {
    case "LOGISTIC":
      for (let i = 0; i < x.length; i++)x[i] = logistic(x[i])
      return x;
    case "RELU":
      for (let i = 0; i < x.length; i++)x[i] = relu(x[i])
      return x;
    case "LEAKY":
      for (let i = 0; i < x.length; i++)x[i] = leaky(x[i])
      return x;
    case "MISH":
      for (let i = 0; i < x.length; i++)x[i] = mish(x[i]);
      return x
    case "SWISH":
      for (let i = 0; i < x.length; i++)x[i] = swish(x[i]);
      return x;
    default: return x;
  }
}

function logistic(x) { return (1 / (1 + Math.exp(-x))); }
function leaky(x) { return ((x > 0) ? x : 0.1 * x); }
function relu(x) { return Math.max(0, x); }
function tanh(x) { return (2 / (1 + Math.exp(-2 * x)) - 1); }
function softplus(x, threshold) {
  if (x > threshold) return x;                    // too large
  else if (x < -threshold) return Math.exp(x);    // too small
  return Math.log(Math.exp(x) + 1);
}
function mish(x) {
  return x * tanh(softplus(x, 20));
}
function swish(x) {
  return x * logistic(x);
}