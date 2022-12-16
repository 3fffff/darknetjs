export function route(options, layers) {
  const lo = "layers" in options ? options["layers"] : 0;
    const groups = "groups" in options ? options["groups"] : 1;
    const group_id = "group_id" in options ? options["group_id"] : 0;
    if (lo == null) new Error("Route Layer must specify input layers");

    const tlayers = lo.split(',')
    const templayers = []
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
    
    return l;
  }