export function route(options) {
  const lo = "layers" in options ? options["layers"] : 0;
  const groups = "groups" in options ? options["groups"] : 1;
  const group_id = "group_id" in options ? options["group_id"] : 0;
  if (lo == null) new Error("Route Layer must specify input layers");

  const tlayers = lo.split(',')
  const sizes = tlayers.length;
  const indexes = []
  for (let i = 0; i < tlayers.length; ++i) {
    let index = parseInt(tlayers[i]);
    if (index < 0) index = options.index + index;
    else index++
    indexes.push(index)
  }
  const l = {};
  l.type = "ROUTE";
  l.index = options.index
  l.batch = options.batch;
  l.input_layers = indexes;
  l.groups = groups
  l.group_id = group_id
  l.outputs = sizes  * options.channels * options.height * options.width * options.batch / groups;
  l.inputs = l.outputs;
  l.output = new Float32Array(l.outputs);
  l.out_w = options.width;
  l.out_h = options.height;
  l.out_c = sizes * options.channels / groups;

  return l;
}