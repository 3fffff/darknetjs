class Backward {
  static convolutional_layer(layers) {
    const l = this
    const delta = layers[l.index + 1].output
    const m = l.n / l.groups;
    const n = l.size * l.size * l.c / l.groups;
    const k = l.out_w * l.out_h;
    const kernel_dim = l.size * l.size * l.c / l.groups;
    const out_wh = l.out_h * l.out_w;

    const col_buffer_data = new Float32Array(kernel_dim * l.out_h * l.out_w);
    const x_offset = l.c / l.groups * l.h * l.w;
    const y_offset = l.out_w * l.out_h * l.out_c / l.batch / l.groups;
    const w_offset = l.filters * kernel_dim / l.groups;
    const out_size = l.out_h * l.out_w;

    Backward.gradient(l.output, l.outputs * l.batch, l.activation, l.delta);
    Backward.bias(l.bias_updates, l.delta, l.batch, l.n, k);
    for (let i = 0; i < l.batch; ++i) {
      for (let j = 0; j < l.groups; ++j) {
        Forward.im2col(
          input.subarray((i * l.groups + j) * x_offset), // input im
          l.c / l.groups,     // input channels
          l.h, l.w,           // input size (h, w)
          l.size, l.size,     // kernel size (h, w)
          l.pad * l.dilation, l.pad * l.dilation,       // padding (h, w)
          l.stride_y, l.stride_x, // stride (h, w)
          l.dilation, l.dilation, // dilation (h, w)
          col_buffer_data);       // output

        Forward.matmul2d(l.weights.subarray(w_offset * (l.groups * b + group)), col_buffer_data, l.output.subarray(y_offset * (l.groups * b + group)), l.filters / l.groups, out_wh, kernel_dim);

        if (delta) {
          Forward.matmul2d(l.weights.subarray(w_offset * (l.groups * b + group)), l.delta.subarray(), delta, l.filters / l.groups, out_wh, kernel_dim);

          Forward.im2col(
            input.workspace,        // input
            l.c / l.groups,         // input channels (h, w)
            l.h, l.w,               // input size (h, w)
            l.size, l.size,         // kernel size (h, w)
            l.pad * l.dilation, l.pad * l.dilation,           // padding (h, w)
            l.stride_y, l.stride_x,     // stride (h, w)
            l.dilation, l.dilation, // dilation (h, w)
            delta + (i * l.groups + j) * (l.c / l.groups) * l.h * l.w); // output (delta)
        }
      }
    }
  }
  static bias(bias_updates, delta, batch, n, size) {
    for (let b = 0; b < batch; ++b)
      for (let i = 0; i < n; ++i)
        bias_updates[i] += sum_array(delta + size * (i + b * n), size);
  }
  static scale_cpu(x_norm, delta, batch, n, size, scale_updates) {
    for (let f = 0; f < n; ++f) {
      let sum = 0;
      for (let b = 0; b < batch; ++b) {
        for (let i = 0; i < size; ++i) {
          const index = i + size * (f + n * b);
          sum += delta[index] * x_norm[index];
        }
      }
      scale_updates[f] += sum;
    }
  }
  static sam_layer(layers) {
    const l = this
    Backward.gradient(l.output, l.outputs * l.batch, l.activation, l.delta);
    const from_output = layers[l.indexs].output;
    const from_delta = layers[l.indexs].delta;

    for (let i = 0; i < l.batch * l.out_c * l.out_w * l.out_h; ++i) {
      l.delta[i] += l.delta[i] * from_output[i]; // l.delta * from  (should be divided by channel_size?)
      from_delta[i] = l.input[i] * l.delta[i]; // input * l.delta
    }
  }
  static route_layer(layers) {
    const l = this
    let offset = 0;
    for (let i = 0; i < l.n; ++i) {
      const index = l.input_layers[i];
      const delta = layers[index].delta;
      const input_size = l.input_sizes[i];
      const part_input_size = input_size / l.groups;
      for (let j = 0; j < l.batch; ++j)
        Backward.axpy_cpu(part_input_size, 1, l.delta + offset + j * l.outputs, 1, delta + j * input_size + part_input_size * l.group_id, 1);
      offset += part_input_size;
    }
  }
  static scale_channels_layer(layers) {
    const l = this
    Backward.gradient(l.output, l.outputs * l.batch, l.activation, l.delta);

    const size = l.batch * l.out_c * l.out_w * l.out_h;
    const channel_size = l.out_w * l.out_h;
    const from_output = layers[l.index + 1].output;
    const from_delta = layers[l.index + 1].delta;

    for (i = 0; i < size; ++i) {
      input.delta[i / channel_size] += l.delta[i] * from_output[i];// / channel_size; // l.delta * from  (should be divided by channel_size?)
      from_delta[i] += input.output[i / channel_size] * l.delta[i]; // input * l.delta
    }
  }
  static shortcut_layer(layers) {
    const l = this
    Backward.gradient(l.output, l.outputs * l.batch, l.activation, l.delta);

    axpy_cpu(l.outputs * l.batch, l.alpha, l.delta, 1, layers[l.index].delta, 1);
    shortcut_cpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta, l.w, l.h, l.c, 1, l.beta, layers[l.index].delta);

  }
  static backward_channel_shuffle_layer(layers) {
    const l = this
    let channel = l.c;
    let group_row = l.groups;
    let batch_size = l.batch;
    let spatial_size = l.w * l.h;
    let feature_map_size = spatial_size * channel;
    let group_column = (let)(channel / group_row);
    for (let n = 0; n < batch_size; ++n) {
      channel_shuffle_op(state.delta + n * feature_map_size, l.delta + n * feature_map_size,
        group_row, group_column, spatial_size);
    }
  }
  static channel_shuffle_op(output, input,  group_row,  group_colomn,  len) {
    for (let i = 0; i < group_row; i++) {
      for (let j = 0; j < group_colomn; j++) {
        float * p_i = input + (i * group_colomn + j) * len;
        float * p_o = output + (j * group_row + i) * len;
        copy_cpu(len, p_i, 1, p_o, 1);
      }
    }
  }

  static avgpool_layer(layers) {
    const l = this
    const delta = layers[l.index + 1]
    for (let b = 0; b < l.batch; ++b) {
      for (let k = 0; k < l.c; ++k) {
        const out_index = k + b * l.c;
        for (let i = 0; i < l.h * l.w; ++i) {
          const in_index = i + l.h * l.w * (k + b * l.c);
          delta[in_index] += l.delta[out_index] / (l.h * l.w);
        }
      }
    }
  }
  static maxpool_layer(layers) {
    const l = this
    const delta = layers[l.index + 1]
    for (let i = 0; i < l.out_h * l.out_w * l.c * l.batch; ++i) {
      const index = l.indexes[i];
      delta[index] += l.delta[i];
    }
  }
  static local_avgpool_layer(layers) {
    const l = this
    const delta = layers[l.index + 1]
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
                if (valid) delta[index] += l.delta[out_index] / (l.size * l.size);
              }
            }
          }
        }
      }
    }
  }
  static dropout_layer(layers) {
    const l = this
    const delta = layers[l.index + 1]
    if (!delta) return;
    for (i = 0; i < l.batch * l.inputs; ++i) {
      const r = l.rand[i];
      if (r < l.probability) delta[i] = 0;
      else delta[i] *= l.scale;
    }
  }
  static forward_dropout_layer(layers) {
    const l = this
    const delta = layers[l.index + 1]
    if (!l.train) return;
    for (i = 0; i < l.batch * l.inputs; ++i) {
      const r = Math.random();
      l.rand[i] = r;
      if (r < l.probability) delta[i] = 0;
      else delta[i] *= l.scale;
    }
  }
  static softmax_layer(layers) {
    const l = this
    const delta = layers[l.index + 1]
    Backward.axpy_cpu(l.inputs * l.batch, 1, l.delta, 1, delta, 1);
  }
  static forward_crop_layer(layers) {
    let count = 0;
    let flip = (l.flip && rand() % 2);
    let dh = rand() % (l.h - l.out_h + 1);
    let dw = rand() % (l.w - l.out_w + 1);
    const scale = 2;
    const trans = -1;
    if (l.noadjust) {
      scale = 1;
      trans = 0;
    }
    if (!state.train) {
      flip = 0;
      dh = (l.h - l.out_h) / 2;
      dw = (l.w - l.out_w) / 2;
    }
    for (let b = 0; b < l.batch; ++b) {
      for (let c = 0; c < l.c; ++c) {
        for (let i = 0; i < l.out_h; ++i) {
          for (let j = 0; j < l.out_w; ++j) {
            if (flip) {
              col = l.w - dw - j - 1;
            } else {
              col = j + dw;
            }
            const row = i + dh;
            const index = col + l.w * (row + l.h * (c + l.c * b));
            l.output[count++] = state.input[index] * scale + trans;
          }
        }
      }
    }
  }
  static axpy_cpu(N, ALPHA, X, INCX, Y, INCY) {
    for (let i = 0; i < N; ++i) Y[i * INCY] += ALPHA * X[i * INCX];
  }
  static gradient(x, a, delta) {
    switch (a) {
      case "LOGISTIC":
        for (let i = 0; i < x.length; i++)delta[i] *= Backward.logistic_gradient(x[i])
        return delta;
      case "RELU":
        for (let i = 0; i < x.length; i++)delta[i] *= Backward.relu_gradient(x[i])
        return delta;
      case "ELU":
        for (let i = 0; i < x.length; i++)delta[i] *= Backward.elu_gradient(x[i])
        return delta;
      case "LEAKY":
        for (let i = 0; i < x.length; i++)delta[i] *= Backward.leaky_gradient(x[i])
        return delta;
    }
  }
  /*  network(network *netp)
   {
 
       network net = *netp;
       let i;
       network orig = net;
       for(i = net.n-1; i >= 0; --i){
           layer l = net.layers[i];
           if(l.stopbackward) break;
           if(i == 0){
               net = orig;
           }else{
               layer prev = net.layers[i-1];
               net.input = prev.output;
               net.delta = prev.delta;
           }
           net.index = i;
           l.backward(l, net);
       }
   }*/

  static relu_gradient(x) { return (x > 0); }
  static elu_gradient(x) { return (x >= 0) + (x < 0) * (x + 1); }
  static leaky_gradient(x) { return (x > 0) ? 1 : .1; }
  static logistic_gradient(x) { return (1 - x) * x; }
  static loggy_gradient(x) {
    const y = (x + 1.0) / 2.0;
    return 2 * (1 - y) * y;
  }
}