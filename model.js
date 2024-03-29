class Model {
  option_find_int(options, name, def) {
    const res = options[name]
    if (res) return Math.floor(parseInt(res))
    return def
  }
  option_find_float(options, name, def) {
    const res = options[name]
    if (res) return parseFloat(res)
    return def
  }
  option_find_str(options, name, def) {
    const res = options[name]
    if (res) return res
    return def
  }
  options_from_layer(options, layer) {
    options.width = layer.out_w
    options.height = layer.out_h
    options.channels = layer.out_c
  }

  parse_convolutional(options, param, quant, wasmSupport) {
    const size = this.option_find_int(options, 'size', 1);
    const filters = this.option_find_int(options, 'filters', 1);
    const pad = this.option_find_int(options, 'pad', 0);
    const padding = pad ? (size >> 1) : this.option_find_int(options, 'padding', 0);
    let stride_x = this.option_find_int(options, 'stride_x', -1);
    let stride_y = this.option_find_int(options, 'stride_y', -1);
    if (stride_x < 1 || stride_y < 1) {
      const stride = this.option_find_int(options, 'stride', 1);
      stride_x = stride_x < 1 ? stride : stride_x;
      stride_y = stride_y < 1 ? stride : stride_y;
    }
    const l = {};
    l.type = "CONVOLUTIONAL";
    l.h = parseInt(param.height); l.w = parseInt(param.width); l.c = parseInt(param.channels);
    l.filters = filters;
    l.groups = this.option_find_int(options, 'groups', 1);
    l.batch = options.batch;
    l.stride_x = stride_x;
    l.stride_y = stride_y;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = this.option_find_int(options, 'batch_normalize', 0);
    l.dilation = 1

    l.weights = new Float32Array(l.c / l.groups * filters * size * size);

    l.biases = new Float32Array(filters);

    l.out_h = Math.floor((l.h + 2 * l.pad - l.size) / l.stride_y) + 1;
    l.out_w = Math.floor((l.w + 2 * l.pad - l.size) / l.stride_x) + 1;
    l.out_c = filters;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = new Float32Array(l.batch * l.outputs);

    l.forward = wasmSupport ? WasmConv : Forward.convolutional_layer
    if (l.batch_normalize != 0) {
      l.scales = new Float32Array(filters);
      for (let i = 0; i < filters; ++i)l.scales[i] = 1;
      l.mean = new Float32Array(filters);
      l.variance = new Float32Array(filters);
    }
    l.activation = this.option_find_str(options, "activation", "logistic").toUpperCase();
    this.options_from_layer(options, l)
    return l;
  }
  parse_deconvolutional(options, param) {
    const size = this.option_find_int(options, 'size', 1);
    const filters = this.option_find_int(options, 'filters', 1);
    const pad = this.option_find_int(options, 'pad', 0);
    const padding = pad ? (size >> 1) : this.option_find_int(options, 'padding', 0);
    let stride_x = this.option_find_int(options, 'stride_x', -1);
    let stride_y = this.option_find_int(options, 'stride_y', -1);
    if (stride_x < 1 || stride_y < 1) {
      const stride = this.option_find_int(options, 'stride', 1);
      stride_x = stride_x < 1 ? stride : stride_x;
      stride_y = stride_y < 1 ? stride : stride_y;
    }
    const l = {};
    l.type = "DECONVOLUTIONAL";
    l.h = parseInt(param.height); l.w = parseInt(param.width); l.c = parseInt(param.channels);
    l.filters = filters;
    l.batch = options.batch;
    l.stride_x = stride_x;
    l.stride_y = stride_y;
    l.size = size;
    l.pad = padding;
    l.dilation = 1

    l.weights = new Float32Array(l.c / l.groups * filters * size * size);

    l.biases = new Float32Array(filters);

    l.out_h = Math.floor((l.h + 2 * l.pad - l.size) / l.stride_y) + 1;
    l.out_w = Math.floor((l.w + 2 * l.pad - l.size) / l.stride_x) + 1;
    l.out_c = filters;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = new Float32Array(l.batch * l.outputs);
    l.forward = Forward.deconvolutional_layer
    l.activation = this.option_find_str(options, "activation", "logistic").toUpperCase();
    this.options_from_layer(options, l)
    return l;
  }

  parse_softmax(options, param) {
    const l = {};
    l.type = 'SOFTMAX';
    l.batch = options.batch;
    l.inputs = param.width * param.height * param.channels;
    l.outputs = l.inputs;
    l.groups = this.option_find_int(options, "groups", 1);
    l.output = new Float32Array(l.inputs * l.batch);
    this.options_from_layer(options, l)
    return l;
  }

  parse_maxpool(options, param, wasmSupport) {
    const stride = this.option_find_int(options, "stride", 1);
    const size = this.option_find_int(options, "size", stride);
    const padding = Math.floor(this.option_find_int(options, "padding", (size - 1)));
    const antialiasing = this.option_find_int(options, 'antialiasing', 0);
    const blur_stride_x = this.option_find_int(options, 'stride_x', stride);
    const blur_stride_y = this.option_find_int(options, 'stride_y', stride);
    const stride_x = antialiasing ? 1 : blur_stride_x;
    const stride_y = antialiasing ? 1 : blur_stride_y;

    const l = {};
    l.type = "MAXPOOL";
    l.batch = options.batch;
    l.h = param.height; l.w = param.width; l.c = param.channels;
    l.pad = padding;
    l.out_w = Math.floor((l.w + padding - size) / stride_x) + 1;
    l.out_h = Math.floor((l.h + padding - size) / stride_y) + 1;
    l.out_c = l.c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.h * l.w * l.c;
    l.size = size;
    l.stride_x = stride_x;
    l.stride_y = stride_y
    l.output = new Float32Array(l.out_h * l.out_w * l.out_c * l.batch);
    l.forward = wasmSupport ? WasmPool : Forward.pool_layer
    this.options_from_layer(options, l)
    return l;
  }
  parse_avgpool(options, param) {
    const l = {};
    l.type = "AVGPOOL";
    l.batch = options.batch;
    l.h = param.height; l.w = param.width; l.c = param.channels;
    l.out_w = 1;
    l.out_h = 1;
    l.out_c = param.channels;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.h * l.w * l.c;
    l.output = new Float32Array(l.out_h * l.out_w * l.out_c * l.batch);
    this.options_from_layer(options, l)
    l.forward = Forward.pool_layer
    return l;
  }

  parse_yolo(options, param) {
    const classes = this.option_find_int(options, "classes", 20);
    const total = this.option_find_int(options, "num", 1);
    let a = this.option_find_str(options, "mask", "0");
    const mask = this.parse_yolo_mask(a, total);
    const max_boxes = this.option_find_int(options, "max", 90);
    const l = {};
    l.type = "YOLO";
    l.filters = total;
    l.total = total;
    l.batch = options.batch;
    l.h = parseInt(param.height);
    l.w = parseInt(param.width);
    l.c = l.total * (classes + 4 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.biases = new Float32Array(total * 2);
    l.mask = new Int32Array(l.total);
    if (mask.Length != 0) l.mask = mask;
    else for (let i = 0; i < l.total; ++i)l.mask[i] = i;
    l.outputs = l.h * l.w * l.total * (classes + 4 + 1);
    l.inputs = l.outputs;
    l.max_boxes = max_boxes;
    l.truths = l.max_boxes * (4 + 1);
    l.output = new Float32Array(l.batch * l.outputs);
    for (let i = 0; i < total * 2; ++i)l.biases[i] = 0.5;
    if (l.outputs != param.inputs) {
      new Error("Error: l.outputs == params.inputs filters= in the [convolutional]-layer doesn't correspond to classes= or mask= in [yolo]-layer ");
    }
    let map_file = this.option_find_str(options, "map", "0");
    if (map_file == null) l.map = read_map(map_file);

    l.jitter = this.option_find_float(options, "jitter", 0.2);

    l.random = this.option_find_int(options, "random", 0);

    a = this.option_find_str(options, "anchors", "0");
    if (a.length != 0) {
      let len = a.length;
      let n = 1;
      for (let i = 0; i < len; ++i) {
        if (a[i] == ',') ++n;
      }
      for (let i = 0; i < n && i < total * 2; ++i) {
        let bias = a.split(',');
        let b = parseFloat(bias[i]);
        l.biases[i] = b;
      }
    }
    l.forward = Forward.YOLODROP
    return l;
  }
  parse_yolo_mask(a, num) {
    if (a.length != 0) {
      let n = 1;
      for (let i = 0; i < a.length; ++i)
        if (a[i] == ',') ++n;
      const mask = new Int32Array(n);
      for (let i = 0; i < n; ++i) {
        let v = a.split(',');
        let vi = v[i].replace(" ", "");
        let val = parseInt(vi);
        mask[i] = val;
      }
      num = n;
      return mask;
    }
    else return null;
  }
  parse_upsample(options, param) {
    let stride = this.option_find_int(options, "stride", 2);
    const l = {};
    l.type = "UPSAMPLE";
    l.tpi = "NEAREST"
    l.batch = options.batch;
    l.w = parseInt(param.width);
    l.h = parseInt(param.height);
    l.c = parseInt(param.channels);
    l.out_w = l.w * stride;
    l.out_h = l.h * stride;
    l.out_c = l.c;
    if (stride < 0) {
      stride = (-1) * stride;
      l.reverse = 1;
      l.out_w = l.w / stride;
      l.out_h = l.h / stride;
    }
    l.stride = stride;
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.inputs = l.w * l.h * l.c;
    l.output = new Float32Array(l.outputs * l.batch);
    l.scale = this.option_find_float(options, "scale", 1);
    if (l.scale == 0) l.scale = 1;
    this.options_from_layer(options, l)
    l.forward = Forward.upsample_layer
    return l;
  }
  parse_dropout(options, param) {
    const probability = this.option_find_float(options, "probability", 0.5);
    const l = {};
    l.type = "DROPOUT";
    l.probability = probability;
    l.inputs = param.inputs;
    l.outputs = param.inputs;
    l.batch = options.batch;
    l.scale = (1.0 / (1.0 - probability));
    l.out_w = parseInt(param.width);
    l.out_h = parseInt(param.height);
    l.out_c = parseInt(param.channels);
    this.options_from_layer(options, l)
    l.forward = Forward.YOLODROP
    return l;
  }

  parse_scale_channels(options, layers) {
    const lo = this.option_find_int(options, "from");
    let index = parseInt(lo);
    if (index < 0) index = options.index + index;
    else index++
    const l = {};
    l.scale_wh = this.option_find_int(options, "scale_wh", 0);
    const from = layers[index];
    l.type = "SCALE_CHANNELS";
    l.batch = options.batch;
    l.w = layers[options.index - 1].out_w;
    l.h = layers[options.index - 1].out_h;
    l.c = layers[options.index - 1].out_c;
    l.out_w = from.out_w;
    l.out_h = from.out_h;
    l.out_c = from.out_c;
    if (l.scale_wh == 0) l.out_c = l.c;
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.inputs = l.outputs;
    l.indexs = from.index;
    l.output = new Float32Array(l.outputs * l.batch);
    l.activation = this.option_find_str(options, "activation", "LINEAR").toUpperCase();
    if (l.activation.toUpperCase() == "SWISH" || l.activation.toUpperCase() == "MISH")
      new Error(" [scale_channels] layer doesn't support SWISH or MISH activations ");
    this.options_from_layer(options, l)
    l.forward = Forward.scale_channels_layer
    return l;
  }

  parse_shortcut(options, layers) {
    const lo = this.option_find_str(options, "from");
    let index = parseInt(lo);
    if (index < 0) index = options.index + index;
    else index++

    const from = layers[index];
    const l = {};
    l.type = "SHORTCUT";
    l.batch = options.batch;
    l.w = from.out_w;
    l.h = from.out_h;
    l.c = from.out_c;
    l.out_w = layers[options.index - 1].out_w;
    l.out_h = layers[options.index - 1].out_h;
    l.out_c = layers[options.index - 1].out_c;
    l.outputs = l.w * l.h * l.c;
    l.inputs = l.outputs;
    l.indexs = from.index;
    l.output = new Float32Array(l.outputs * l.batch);
    this.options_from_layer(options, l)
    l.activation = options.activation.toUpperCase();
    l.forward = Forward.shortcut_layer
    return l;
  }
  parse_route(options, layers) {
    let lo = this.option_find_str(options, "layers");
    const groups = this.option_find_int(options, "groups", 1);
    const group_id = this.option_find_int(options, "group_id", 0);
    if (lo == null) new Error("Route Layer must specify input layers");

    const tlayers = lo.split(',')
    const templayers = Array()
    const sizes = Array(tlayers.length);
    for (let i = 0; i < tlayers.length; ++i) {
      let index = parseInt(tlayers[i]);
      if (index < 0) index = options.index + index;
      else index++
      templayers.push(layers[index])
      sizes[i] = layers[index].outputs;
    }

    const l = {};
    l.type = "ROUTE";
    l.batch = options.batch;
    l.filters = templayers.length;
    l.input_layers = templayers;
    l.input_sizes = sizes;
    l.groups = groups
    l.group_id = group_id
    let outputs = 0
    for (let i = 0; i < l.filters; ++i) outputs += sizes[i];
    l.outputs = outputs / groups;
    l.inputs = l.outputs;
    l.output = new Float32Array(l.outputs * options.batch);
    let first = templayers[0];
    l.out_w = first.out_w;
    l.out_h = first.out_h;
    l.out_c = first.out_c / groups;
    for (let i = 1; i < templayers.length; ++i) {
      let next = templayers[i];
      if (next.out_w == first.out_w && next.out_h == first.out_h)
        l.out_c += next.out_c;
      else
        l.out_h = l.out_w = l.out_c = 0;
    }
    this.options_from_layer(options, l)
    l.forward = Forward.route_layer
    return l;
  }

  parse_sam(options, layers) {
    const lo = this.option_find_str(options, "from");
    let index = parseInt(lo);
    if (index < 0) index = options.index + index;
    else index++
    let from = layers[index];

    const l = {}
    l.type = "SAM";
    l.batch = options.batch;
    l.w = layers[options.index - 1].out_w;
    l.h = layers[options.index - 1].out_h;
    l.c = layers[options.index - 1].out_c;

    l.out_w = from.out_w;
    l.out_h = from.out_h;
    l.out_c = from.out_c;

    l.outputs = l.out_w * l.out_h * l.out_c;
    l.inputs = l.outputs;
    l.indexs = from.index;
    l.input_layers = templayers;

    l.output = new Float32Array(l.outputs * options.batch);
    l.activation = this.option_find_str(options, "activation", "LINEAR").toUpperCase();
    if (l.activation == 'SWISH' || l.activation == 'MISH')
      console.log(" [sam] layer doesn't support SWISH or MISH activations \n");
    this.options_from_layer(options, l)
    l.forward = Forward.sam_layer
    return l;
  }
  parse_connected_layer(options, param) {
    const total_batch = batch * steps;
    const l = {};
    l.type = "CONNECTED";

    l.inputs = param.inputs;
    l.outputs = param.outputs;
    l.batch = param.batch;
    l.batch_normalize = this.option_find_int(options, 'batch_normalize', 0);
    l.h = 1;
    l.w = 1;
    l.c = inputs;
    l.out_h = 1;
    l.out_w = 1;
    l.out_c = outputs;
    l.n = l.out_c;
    l.size = 1;
    l.stride = l.stride_x = l.stride_y = 1;
    l.pad = 0;

    l.learning_rate_scale = 1;

    l.output = new Float32Array(total_batch * l.outputs);

    l.weights = new Float32Array(l.outputs * l.inputs);
    l.biases = new Float32Array(l.outputs);
    l.forward = Forward.connected_layer
    if (l.batch_normalize != 0) {
      l.scales = new Float32Array(n);
      for (let i = 0; i < n; ++i)l.scales[i] = 1;
      l.mean = new Float32Array(n);
      l.variance = new Float32Array(n);
    }
    l.activation = this.option_find_str(options, "activation", "logistic").toUpperCase();
    this.options_from_layer(options, l)
    return l;
  }

  cost_layer(options, param) {
    const l = {};
    l.type = "COST";

    l.scale = param.scale;
    l.batch = param.batch;
    l.inputs = param.inputs;
    l.outputs = param.inputs;
    l.cost_type = param.cost_type;
    l.delta = new Float32Array(inputs * batch);
    l.output = new Float32Array(inputs * batch);
    l.cost = 0;
    this.options_from_layer(options, l)
    return l;
  }

  crop_layer(options, param) {
    const l = {};
    l.type = "CROP";
    l.batch = param.batch;
    l.h = param.h;
    l.w = param.w;
    l.c = param.c;
    l.scale = param.crop_height / l.h;
    l.flip = param.flip;
    l.angle = param.angle;
    l.saturation = param.saturation;
    l.exposure = param.exposure;
    l.out_w = param.crop_width;
    l.out_h = param.crop_height;
    l.out_c = c;
    l.inputs = l.w * l.h * l.c;
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.output = new Float32Array(l.outputs * batch);
    this.options_from_layer(options, l)
    return l;
  }

  read_cfg(cfg) {
    const lines = cfg.split("\n");
    const sections = [];
    let section = null;
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].replace(/\s/g, '');
      if (line.length > 0) {
        switch (line[0]) {
          case '#':
          case ';':
            break;
          case '[': {
            section = {};
            section.line = i;
            section.type = line[line.length - 1] === ']' ? line.substring(1, line.length - 1) : line.substring(1);
            section.options = {};
            sections.push(section);
            break;
          }
          default: {
            if (!section || line[0] < 0x20 || line[0] > 0x7E) throw new Error("Invalid cfg '" + text.replace(/[^\x20-\x7E]+/g, '').trim() + "' at line " + i.toString() + ".");
            if (section) {
              const index = line.indexOf('=');
              if (index < 0) throw new Error("Invalid cfg '" + text.replace(/[^\x20-\x7E]+/g, '').trim() + "' at line " + i.toString() + ".");
              const key = line.substring(0, index);
              const value = line.substring(index + 1);
              section.options[key] = value;
            }
            break;
          }
        }
      }
    }
    return sections;
  }
  load_weights_upto_cpu(layers, data, merge, quant) {
    let offset = 0, counter = 0
    const major = data.getInt32(offset, true);
    offset += 4
    const minor = data.getInt32(offset, true);
    offset += 8
    offset = ((major * 10 + minor) >= 2) ? offset + 8 : offset + 4
    for (let j = 1; j < layers.length; ++j) {
      if (layers[j].dontload != 0) continue;
      if (layers[j].type == "CONVOLUTIONAL" || layers[j].type == "DECONVOLUTIONAL" || layers[j].type == "CONNECTED") {
        const l = layers[j]
        const num = l.filters * l.c / l.groups * l.size * l.size;
        for (let i = 0; i < l.filters; i++) { l.biases[i] = data.getFloat32(offset, true); offset += 4 }
        if (l.batch_normalize == 1) {
          for (let i = 0; i < l.filters; i++) { l.scales[i] = data.getFloat32(offset, true); offset += 4 }
          for (let i = 0; i < l.filters; i++) { l.mean[i] = data.getFloat32(offset, true); offset += 4 }
          for (let i = 0; i < l.filters; i++) { l.variance[i] = data.getFloat32(offset, true); offset += 4 }
        }
        for (let i = 0; i < num; i++) { l.weights[i] = data.getFloat32(offset, true); offset += 4; }
        if (l.batch_normalize == 1 && merge == 1) {
          for (let f = 0; f < l.filters; ++f) {
            l.biases[f] = l.biases[f] - l.scales[f] * l.mean[f] / (Math.sqrt(l.variance[f] + 0.00001));
            const precomputed = l.scales[f] / (Math.sqrt(l.variance[f] + 0.00001));
            const filter_size = l.size * l.size * l.c / l.groups;
            for (let i = 0; i < filter_size; ++i) l.weights[f * filter_size + i] *= precomputed;
          }
          l.scales = null; l.mean = null; l.variance = null;
          l.batch_normalize = 0;
        }

        if (l.type === "CONVOLUTIONAL" && l.batch_normalize === 0 && quant) {
          l.input_quant_multipler = (counter < layers[0].input_calibration.length) ? layers[0].input_calibration[counter] : 40;
          counter++;
          l.weights_quant_multipler = this.get_multiplier(l.weights, 8) / 4;    // good [2 - 8], best 4
          l.weights_quant = new Int8Array(l.weights.length)
          l.biases_quant = new Int8Array(l.biases.length)
          const num = l.c / l.groups * l.size * l.size;
          for (let fil = 0; fil < num; ++fil) {
            for (let i = 0; i < l.filters; ++i) {
              const w = l.weights[fil * l.filters + i] * l.weights_quant_multipler;
              l.weights_quant[fil * l.filters + i] = Math.max(Math.abs(w), 127);
            }
          }
          const R_MULT = 32
          l.output_multipler = l.input_quant_multipler / (l.weights_quant_multipler * l.input_quant_multipler / R_MULT);
          l.ALPHA1 = 1 / (l.input_quant_multipler * l.weights_quant_multipler);
          for (let fil = 0; fil < num; ++fil) {
            const biases_multipler = l.output_multipler * l.weights_quant_multipler * l.input_quant_multipler / R_MULT;
            l.biases_quant[fil] = l.biases[fil] * biases_multipler;
          }
        }
      }
    }
  }

  get_multiplier(arr, bits_length) {
    const number_of_ranges = 32;
    const start_range = 1 / 65536;
    //distribution
    const count = new Int16Array(number_of_ranges);
    for (let i = 0; i < arr.length; ++i) {
      let cur_range = start_range;
      for (let j = 0; j < number_of_ranges; ++j) {
        if (Math.abs(cur_range) <= arr[i] && arr[i] < Math.abs(cur_range * 2))
          count[j]++;
        cur_range *= 2;
      }
    }

    let max_count_range = 0;
    let index_max_count = 0;
    for (let j = 0; j < number_of_ranges; ++j) {
      let counter = 0;
      for (let i = j; i < (j + bits_length) && i < number_of_ranges; ++i)
        counter += count[i];
      if (max_count_range < counter) {
        max_count_range = counter;
        index_max_count = j;
      }
    }
    let multiplier = 1 / (start_range * Math.pow(2.0, index_max_count));
    return multiplier;
  }

  parse_network_cfg(filename, weights, quant, wasm, webgl) {
    let sections = this.read_cfg(filename);
    const layers = [];
    if (sections[0].type == "[net]" && sections[0].type == "[network]") throw new Error("First section must be [net] or [network]");
    let net = {}
    net.w = this.option_find_int(sections[0].options, "width", 1);
    net.h = this.option_find_int(sections[0].options, "height", 1);
    net.c = this.option_find_int(sections[0].options, "channels", 1);
    net.batch = this.option_find_int(sections[0].options, "batch", 1);
    net.train = this.option_find_int(sections[0].options, "train", false);
    net.quant = quant
    net.wasmSupport = wasm
    if (quant) {
      const input_calibration = sections[0].options.input_calibration.split(',')
      net.input_calibration = new Float32Array(input_calibration.length)
      for (let i = 0; i < input_calibration.length; i++)
        net.input_calibration[i] = parseFloat(input_calibration[i])
    }
    layers.push(net)
    for (let i = 1; i < sections.length; i++) {
      let l = {};
      sections[i].options.index = i
      sections[i].options.batch = net.batch
      if (sections[i].type.toUpperCase() == 'CONVOLUTIONAL') l = this.parse_convolutional(sections[i].options, sections[i - 1].options, net.quant, net.wasmSupport);
      else if (sections[i].type.toUpperCase() == 'DECONVOLUTIONAL') l = this.parse_deconvolutional(sections[i].options, sections[i - 1].options);
      else if (sections[i].type.toUpperCase() == 'CONNECTED') l = this.parse_connected(sections[i].options, sections[i - 1].options);
      else if (sections[i].type.toUpperCase() == 'SOFTMAX') l = this.parse_softmax(sections[i].options, sections[i - 1].options);
      else if (sections[i].type.toUpperCase() == 'MAXPOOL') l = this.parse_maxpool(sections[i].options, sections[i - 1].options, net.wasmSupport);
      else if (sections[i].type.toUpperCase() == 'AVGPOOL') l = this.parse_avgpool(sections[i].options, sections[i - 1].options);
      else if (sections[i].type.toUpperCase() == 'ROUTE') l = this.parse_route(sections[i].options, layers);
      else if (sections[i].type.toUpperCase() == 'YOLO') l = this.parse_yolo(sections[i].options, sections[i - 1].options);
      else if (sections[i].type.toUpperCase() == 'SHORTCUT') l = this.parse_shortcut(sections[i].options, layers);
      else if (sections[i].type.toUpperCase() == 'DROPOUT') l = this.parse_dropout(sections[i].options, sections[i - 1].options);
      else if (sections[i].type.toUpperCase() == 'UPSAMPLE') l = this.parse_upsample(sections[i].options, sections[i - 1].options);
      else if (sections[i].type.toUpperCase() == 'SCALE_CHANNELS') l = this.parse_scale_channels(sections[i].options, layers);
      else if (sections[i].type.toUpperCase() == 'SAM') l = this.parse_sam(sections[i].options, layers);
      else throw new Error("Type not recognized: " + sections[i].type);
      l.dontload = this.option_find_int(sections[i].options, "dontload", 0);
      l.dontloadscales = this.option_find_int(sections[i].options, "dontloadscales", 0);
      l.index = i
      if (!l.type) continue
      layers.push(l);
    }
    this.load_weights_upto_cpu(layers, weights, 1, net.quant)
    if (webgl) this.initGL(layers)
    console.log(layers)
    return layers
  }
  initGL(layers) {
    layers[0].webgl = new WebGL("webgl2")
    for (let i = 1; i < layers.length; i++) {
      const l = layers[i]
      if (l.type.toUpperCase() == 'CONVOLUTIONAL') {
        const textures = [{ index: l.index, activation: l.activation, TextureID: "t" + (l.index - 1), activation: l.activation, pad: l.pad, size: l.size, shape: [l.batch, l.c, l.h, l.w] },
        { batch: l.batch, filters: l.filters, size: l.size, output: l.weights, dilation: l.dilation, TextureID: 'w' + l.index, groups: l.groups, shape: [l.filters, l.c, l.size, l.size], pad: l.pad, stride_x: l.stride_x, stride_y: l.stride_y },
        { output: l.biases, TextureID: 'bias' + l.index, shape: [l.filters] }]
        l.artifacts = []
        if (l.groups == 1) {
          const glProg = WebGLConv.createProgramInfos(webgl, textures, [l.batch, l.out_c, l.out_h, l.out_w], l.activation)
          l.artifacts.push(webgl.programManager.build(glProg[0]));
          l.artifacts.push(webgl.programManager.build(glProg[1]));
          if (glProg[2]) l.artifacts.push(webgl.programManager.build(glProg[2]));
          l.runData = WebGLConv.createRunDatas(webgl, textures, glProg, l.index, l.activation)
        }
        else {
          const glProg = WebGLGroupConv.createProgramInfos(webgl, textures, [l.batch, l.out_c, l.out_h, l.out_w], l.activation)
          l.artifacts.push(webgl.programManager.build(glProg[0]));
          if (glProg[1]) l.artifacts.push(webgl.programManager.build(glProg[1]));
          l.runData = WebGLGroupConv.createRunDatas(webgl, textures, glProg, l.index, l.activation)
        }
      }
      else if (l.type.toUpperCase() == 'MAXPOOL' || l.type.toUpperCase() == 'LOCALAVG' || l.type.toUpperCase() == 'AVGPOOL') {
        const textures = [{ TextureID: "t" + (l.index - 1), pad: l.pad, size: l.size, stride_x: l.stride_x, stride_y: l.stride_x, shape: [l.batch, l.c, l.h, l.w] }]
        const glProg = WebGLPool.createProgramInfo(webgl, textures[0], [l.batch, l.out_c, l.out_h, l.out_w], l.type)
        l.artifacts = [webgl.programManager.build(glProg)]
        l.runData = WebGLPool.createRunData(webgl, textures, glProg, l.index)
      }
      else if (l.type.toUpperCase() == 'UPSAMPLE') {
        const textures = [{ TextureID: "t" + (l.index - 1), scale: 1, stride: l.stride, shape: [l.batch, l.c, l.h, l.w] }]
        const glProg = WebGLUpsample.createProgramInfo(webgl, textures[0], [l.batch, l.out_c, l.out_h, l.out_w])
        l.artifacts = [webgl.programManager.build(glProg)];
        l.runData = WebGLUpsample.createRunData(webgl, textures[0], glProg, l.index)
      }
      else if (l.type.toUpperCase() == 'ROUTE') {
        const textures = []
        let glProg;
        if (l.input_layers.length == 1) {
          textures.push({ groups: 0, TextureID: "t" + l.input_layers[0].index, shape: [l.input_layers[0].batch, l.out_c, l.input_layers[0].h, l.input_layers[0].w] })
          glProg = WebGLRoute.createSplitProgramInfo(webgl, textures, [l.batch, l.out_c, l.out_h, l.out_w], l.group_id * l.out_c)
        } else {
          for (let i = 0; i < l.input_layers.length; ++i) textures.push({ groups: l.groups, TextureID: "t" + l.input_layers[i].index, shape: [l.input_layers[i].batch, l.input_layers[i].out_c, l.input_layers[i].out_h, l.input_layers[i].out_w] })
          glProg = WebGLRoute.createProgramInfo(webgl, textures, [l.batch, l.out_c, l.out_h, l.out_w])
        }
        l.artifacts = [webgl.programManager.build(glProg)];
        l.runData = WebGLRoute.createRunData(webgl, textures, glProg, l.index)
      }
      else if (l.type.toUpperCase() == 'CONNECTED') {
        const textures = [{ TextureID: "tw" + l.index, shape: [l.weights.length, 1], output: l.weights }, { TextureID: "t" + (l.index - 1), shape: [1, l.weights.length] }]
        const glProg = WebGLMatMul.createProgramInfo(webgl, l)
        l.artifacts = [webgl.programManager.build(glProg)];
        l.runData = WebGLMatMul.createRunData(webgl, textures, glProg, l.index)
      }
      else if (l.type.toUpperCase() == 'SHORTCUT' || l.type.toUpperCase() == 'SAM') {
        const textures = [{ TextureID: "t" + (l.index - 1), shape: [l.batch, l.c, l.h, l.w] }, { TextureID: "t" + l.indexs, shape: [l.batch, l.c, l.h, l.w] }]
        const glProg = WebGLSum.createProgramInfo(webgl, textures, [l.batch, l.out_c, l.out_h, l.out_w], l.type.toUpperCase())
        l.artifacts = [webgl.programManager.build(glProg)];
        l.runData = WebGLSum.createRunData(webgl, textures, glProg, l.index)
      }
      else if (l.type.toUpperCase() == 'SCALE_CHANNELS') {
        const textures = [{ TextureID: "t" + (l.index - 1), shape: [l.batch, l.c, l.h, l.w] }, { TextureID: "t" + l.indexs, shape: [l.batch, l.c, l.h, l.w] }]
        const glProg = WebGLSum.createScaleChannelsProgramInfo(webgl, textures, [l.batch, l.out_c, l.out_h, l.out_w], l.type.toUpperCase())
        l.artifacts = [webgl.programManager.build(glProg)];
        l.runData = WebGLSum.createRunData(webgl, textures, glProg, l.index)
      }
      else if (l.type.toUpperCase() == 'DROPOUT') {
        l.artifacts = layers[i - 1].artifacts
        l.runData = layers[i - 1].runData
        layers[0].webgl.setTextureData("t" + l.index, l.runData[l.runData.length - 1].outputTextureData);
      }
      else if (l.type.toUpperCase() == 'YOLO') {
        l.forwardGl = forwardWebglYolo
        continue
      }
      l.forwardGl = forwardWebgl
    }
  }
}
