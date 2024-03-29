class Forward {
  static im2col(data_im, data_col, channels, height,
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
  static matmul(A, B, C, M, N, K) {
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
  static BatchActivate(l, out_size) {
    for (let nc = 0; nc < l.batch * l.filters; nc++) {
      const offset = nc * out_size;
      if (l.batch_normalize == 1)
        for (let i = 0; i < out_size; i++)l.output[offset + i] = l.scales[nc % l.filters] * ((l.output[offset + i] - l.mean[nc % l.filters]) / Math.sqrt(l.variance[nc % l.filters] + 0.000001)) + l.biases[nc % l.filters];
      else if (l.biases.length != 0) for (let i = 0; i < out_size; i++)l.output[offset + i] += l.biases[nc % l.filters];
    }
    l.output = Forward.activate(l.output, l.activation);
  }
  static convolutional_layer(layers) {
    const l = this
    const input = layers[l.index - 1].output
    const kernel_dim = l.size * l.size * l.c / l.groups;
    const out_wh = l.out_h * l.out_w;

    const col_buffer_data = new Float32Array(kernel_dim * l.out_h * l.out_w);
    const x_offset = l.c * l.h * l.w / l.groups;
    const y_offset = l.out_w * l.out_h * l.out_c / l.groups;
    const w_offset = l.filters * kernel_dim / l.groups;
    const out_size = l.out_h * l.out_w;
    for (let b = 0; b < l.batch; ++b) {
      for (let group = 0; group < l.groups; ++group) {
        Forward.im2col(input.subarray(x_offset * group), col_buffer_data, l.c / l.groups, l.h, l.w, l.size, l.size, l.dilation, l.dilation, l.pad, l.pad, l.pad, l.pad, l.stride_y, l.stride_x);
        Forward.matmul(l.weights.subarray(w_offset * group), col_buffer_data, l.output.subarray(y_offset * group), l.filters / l.groups, out_wh, kernel_dim);
      }
    }
    Forward.BatchActivate(l, out_size)
  }

  static convolutional_layer_q(layers) {
    const l = this
    const input = l.index - 1 == 0 ? layers[l.index - 1].output : new Int8Array(layers[l.index - 1].output.length)
    if (l.index - 1 != 0) {
      for (let i = 0; i < layers[l.index - 1].output.length; ++i) {
        const src = layers[l.index - 1].output[i] * l.input_quant_multipler;
        input[i] = Math.max(Math.abs(src), 127);
      }
    }
    const kernel_dim = l.size * l.size * l.c / l.groups;
    const out_wh = l.out_h * l.out_w;

    const col_buffer_data = new Int8Array(kernel_dim * l.out_h * l.out_w);
    const x_offset = l.c * l.h * l.w / l.groups;
    const y_offset = l.out_w * l.out_h * l.out_c / l.groups;
    const w_offset = l.filters * kernel_dim / l.groups;
    const output_q = new Int32Array(l.output.length)
    for (let b = 0; b < l.batch; ++b) {
      for (let group = 0; group < l.groups; ++group) {
        Forward.im2col(input.subarray(x_offset * group), col_buffer_data, l.c / l.groups, l.h, l.w, l.size, l.size, l.dilation, l.dilation, l.pad, l.pad, l.pad, l.pad, l.stride_y, l.stride_x);
        Forward.matmul(l.weights_quant.subarray(w_offset * group), col_buffer_data, output_q.subarray(y_offset * group), l.filters / l.groups, out_wh, kernel_dim);
      }
    }
    for (let i = 0; i < l.outputs; ++i) l.output[i] = output_q[i] * l.ALPHA1;
    Forward.BatchActivate(l, out_size)
  }

  static deconvolutional_layer(layers) {
    const l = this
    const input = layers[l.index - 1].output
    const kernel_dim = l.size * l.size * l.c / l.groups;
    const out_wh = l.out_h * l.out_w;

    const col_buffer_data = new Float32Array(kernel_dim * l.out_h * l.out_w);
    const x_offset = l.c / l.groups * l.h * l.w;
    const y_offset = l.out_w * l.out_h * l.out_c / l.groups;
    const w_offset = l.filters * kernel_dim / l.groups;
    const out_size = l.out_h * l.out_w;
    for (let b = 0; b < l.batch; ++b) {
      for (let group = 0; group < l.groups; ++group) {
        Forward.matmul(l.weights.subarray(w_offset * group), input.subarray(x_offset * group), col_buffer_data, l.filters / l.groups, out_wh, kernel_dim);
        Forward.im2col(col_buffer_data, l.output.subarray(y_offset * group), l.out_c / l.groups, l.out_h, l.out_w, l.size, l.size, l.dilation, l.dilation, l.pad, l.pad, l.pad, l.pad, l.stride_y, l.stride_x);
      }
    }
    Forward.BatchActivate(l, out_size)
  }
  static connected_layer(layers) {
    const l = this
    const input = layers[l.index - 1].output
    Forward.matmul(input, l.weights, l.output, l.batch, l.inputs, l.outputs);
    Forward.BatchActivate(l, l.outputs)
  }
  static pool_layer(layers) {
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
  static route_layer(layers) {
    const l = this
    let offset = 0;
    for (let i = 0; i < l.input_layers.length; ++i) {
      const index = l.input_layers[i].index;                  // source layer index
      const input = layers[index].output;  // source layer output ptr
      const input_size = l.input_sizes[i];              // source layer size
      const part_input_size = input_size / l.groups;
      for (let j = 0; j < l.batch; ++j)l.output.set(input.subarray(j * input_size + part_input_size * l.group_id, j * input_size + part_input_size * l.group_id + part_input_size), offset + j * l.outputs);
      offset += part_input_size;
    }
  }
  static upsample_layer(layers) {
    const l = this
    const input = layers[l.index - 1].output
    Forward.upsampleNearest(l, input)
  }
  static upsampleNearest(l, input, forward = true) {
    for (let b = 0; b < l.batch; ++b)
      for (let k = 0; k < l.c; ++k)
        for (let j = 0; j < l.h * l.stride; ++j)
          for (let i = 0; i < l.w * l.stride; ++i)
            if (forward) l.output[b * l.w * l.h * l.c * l.stride * l.stride + k * l.w * l.h * l.stride * l.stride + j * l.w * l.stride + i] =
              l.scale * input[b * l.w * l.h * l.c + k * l.w * l.h + ~~(j / l.stride) * l.w + ~~(i / l.stride)];
            else input[b * l.w * l.h * l.c + k * l.w * l.h + ~~(j / l.stride) * l.w + ~~(i / l.stride)] =
              l.scale * l.output[b * l.w * l.h * l.c * l.stride * l.stride + k * l.w * l.h * l.stride * l.stride + j * l.w * l.stride + i];
  }

  static softmax_layer(layers) {
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
  static scale_channels_layer(layers) {
    for (let i = 0; i < this.output.length; ++i)this.output[i] = layers[this.index - 1].output[~~(i / (this.out_w * this.out_h))] * layers[this.indexs].output[i];
    this.output = Forward.activate(this.output, this.activation);
  }

  static shortcut_layer(layers) {
    for (let i = 0; i < this.output.length; ++i)this.output[i] = layers[this.index - 1].output[i] + layers[this.indexs].output[i];
    this.output = Forward.activate(this.output, this.activation);
  }
  static sam_layer(layers) {
    for (let i = 0; i < this.output.length; ++i)this.output[i] = layers[this.index - 1].output[i] * layers[this.indexs].output[i];
    this.output = Forward.activate(this.output, this.activation);
  }

  static YOLODROP(layers) {
    this.output = layers[this.index - 1].output;
  }

  static logistic(x) { return (1 / (1 + Math.exp(-x))); }
  static leaky(x) { return ((x > 0) ? x : 0.1 * x); }
  static relu(x) { return Math.max(0, x); }
  static tanh(x) { return (2 / (1 + Math.exp(-2 * x)) - 1); }
  static softplus(x, threshold) {
    if (x > threshold) return x;                    // too large
    else if (x < -threshold) return Math.exp(x);    // too small
    return Math.log(Math.exp(x) + 1);
  }
  static mish(x) {
    return x * Forward.tanh(Forward.softplus(x, 20));
  }
  static swish(x) {
    return x * Forward.logistic(x);
  }
  static activate(x, a) {
    switch (a) {
      case "LOGISTIC":
        for (let i = 0; i < x.length; i++)x[i] = Forward.logistic(x[i])
        return x;
      case "RELU":
        for (let i = 0; i < x.length; i++)x[i] = Forward.relu(x[i])
        return x;
      case "LEAKY":
        for (let i = 0; i < x.length; i++)x[i] = Forward.leaky(x[i])
        return x;
      case "MISH":
        for (let i = 0; i < x.length; i++)x[i] = Forward.mish(x[i]);
        return x
      case "SWISH":
        for (let i = 0; i < x.length; i++)x[i] = Forward.swish(x[i]);
        return x;
      default: return x;
    }
  }
}
