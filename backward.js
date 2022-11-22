class Backward {
  static convolutional_layer(layers) {
    const l = this
    const kernel_dim = l.size * l.size * l.c / l.groups;
    const out_wh = l.out_h * l.out_w;

    const col_buffer_data = new Float32Array(kernel_dim * l.out_h * l.out_w);
    const x_offset = l.c / l.groups * l.h * l.w;
    const y_offset = l.out_w * l.out_h * l.out_c / l.groups;
    const w_offset = l.filters * kernel_dim / l.groups;

    Backward.gradient(l.output, l.outputs * l.batch, l.activation, l.delta);
    Backward.bias(l.bias_updates, l.delta, l.batch, l.n, k);
    for (let i = 0; i < l.batch; ++i) {
      for (let j = 0; j < l.groups; ++j) {
        Forward.im2col(input.subarray(x_offset * group), col_buffer_data, l.c / l.groups, l.h, l.w, l.size, l.size, l.dilation, l.dilation, l.pad, l.pad, l.pad, l.pad, l.stride_y, l.stride_x);
        Forward.matmul(l.weights.subarray(w_offset * group), col_buffer_data, l.output.subarray(y_offset * group), l.filters / l.groups, out_wh, kernel_dim);
        if (layers[l.index - 1].delta) {
          Forward.im2col(layers[l.index - 1].delta.subarray(x_offset * group), col_buffer_data, l.c / l.groups, l.h, l.w, l.size, l.size, l.dilation, l.dilation, l.pad, l.pad, l.pad, l.pad, l.stride_y, l.stride_x);
          Forward.matmul(l.weights.subarray(w_offset * group), col_buffer_data, l.delta.subarray(y_offset * group), l.filters / l.groups, out_wh, kernel_dim);
        }
      }
    }
  }
  static bias(bias_updates, delta, batch, n, size) {
    for (let b = 0; b < batch; ++b)
      for (let i = 0; i < n; ++i)
        bias_updates[i] += sum_array(delta.subarray(size * (i + b * n)), size);
  }
  static scale(x_norm, delta, batch, n, size, scale_updates) {
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
      layers[l.index - 1].delta[i] += l.delta[i] * from_output[i]; // l.delta * from  (should be divided by channel_size?)
      from_delta[i] = input[i] * l.delta[i]; // input * l.delta
    }
  }
  static route_layer(layers) {
    const l = this
    let offset = 0;
    for (let i = 0; i < l.input_layers.length; ++i) {
      const index = l.input_layers[i].index;                  // source layer index
      const delta = layers[index].delta;  // source layer output ptr
      const input_size = l.input_sizes[i];              // source layer size
      const part_input_size = input_size / l.groups;
      for (let j = 0; j < l.batch; ++j)delta.set(l.delta.subarray(j * input_size + part_input_size * l.group_id, j * input_size + part_input_size * l.group_id + part_input_size), offset + j * l.outputs);
      offset += part_input_size;
    }
  }
  static scale_channels_layer(layers) {
    const l = this
    Backward.gradient(l.output, l.outputs * l.batch, l.activation, l.delta);

    const size = l.batch * l.out_c * l.out_w * l.out_h;
    const channel_size = l.out_w * l.out_h;
    const from_output = layers[l.index - 1].output;
    const from_delta = layers[l.index - 1].delta;

    for (i = 0; i < size; ++i) {
      layers[l.index].delta[i / channel_size] += l.delta[i] * from_output[i];// / channel_size; // l.delta * from  (should be divided by channel_size?)
      from_delta[i] += input[i / channel_size] * l.delta[i]; // input * l.delta
    }
  }
  static shortcut_layer(layers) {
    const l = this
    Backward.gradient(l.output, l.outputs * l.batch, l.activation, l.delta);
    for (let i = 0; i < l.outputs * l.batch; ++i) layers[i-1].delta[i] +=  l.delta[i];
    shortcut_cpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta, l.w, l.h, l.c, 1, l.beta, layers[l.index].delta);

  }

  static avgpool_layer(layers) {
    const l = this
    for (let b = 0; b < l.batch; ++b) {
      for (let k = 0; k < l.c; ++k) {
        const out_index = k + b * l.c;
        for (let i = 0; i < l.h * l.w; ++i) {
          const in_index = i + l.h * l.w * (k + b * l.c);
          layers[l.index - 1].delta[in_index] += l.delta[out_index] / (l.h * l.w);
        }
      }
    }
  }
  static maxpool_layer(layers) {
    const l = this
    for (let i = 0; i < l.out_h * l.out_w * l.c * l.batch; ++i) {
      const index = l.indexes[i];
      layers[l.index - 1].delta[index] += l.delta[i];
    }
  }
  static local_avgpool_layer(layers) {
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
  static dropout_layer(layers) {
    const l = this
    if (!delta) return;
    for (i = 0; i < l.batch * l.inputs; ++i) {
      const r = l.rand[i];
      if (r < l.probability) delta[i] = 0;
      else layers[l.index - 1].delta[i] *= l.scale;
    }
  }
  static forward_dropout_layer(layers) {
    const l = this
    if (!l.train) return;
    for (i = 0; i < l.batch * l.inputs; ++i) {
      const r = Math.random();
      l.rand[i] = r;
      if (r < l.probability) layers[l.index - 1].delta[i] = 0;
      else layers[l.index - 1].delta[i] *= l.scale;
    }
  }
  static softmax_layer(layers) {
    const l = this
    for (let i = 0; i < l.inputs * l.batch; ++i) layers[l.index - 1].delta[i] += l.delta[i];
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
