export function route(options) {
  const lo = "layers" in options ? options["layers"] : 0;
  const groups = "groups" in options ? options["groups"] : 1;
  const group_id = "group_id" in options ? options["group_id"] : 0;
  if (lo == null) throw new Error("Route Layer must specify input layers");
  const l = {};
  l.type = "ROUTE";
  l.index = options.index
  l.batch = options.batch;

  l.groups = groups
  l.group_id = group_id

  l.out_w = options.width;
  l.out_h = options.height;
  let out_c = 0
  const tlayers = lo.split(',')
  const indexes = []
  for (let i = 0; i < tlayers.length; ++i) {
    let index = parseInt(tlayers[i]);
    if (index < 0) index = options.index + index;
    else index++
    indexes.push(index)
    out_c += options.route_channels[i]
  }
  l.out_c = out_c / groups;
  l.input_layers = indexes;
  l.route_channels = options.route_channels
  l.outputs = out_c * options.height * options.width * options.batch / groups;
  l.inputs = l.outputs;
  l.output = new Float32Array(l.outputs);
  return l;
}