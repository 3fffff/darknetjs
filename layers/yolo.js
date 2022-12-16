export function yolo(options, param) {
  const classes = "classes" in options ? options['classes'] : 20;
  const total = "num" in options ? options["num"] : 1;
  let a = "mask" in options ? options["mask"] : 1;
  const mask = parse_yolo_mask(a);
  const max_boxes = "max" in options ? options["max"] : 1;
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
  let map_file = "map" in options ? options["map"] : 0;
  if (map_file == null) l.map = read_map(map_file);

  l.jitter = "jitter" in options ? options["jitter"] : 0.2;

  l.random = "random" in options ? options["random"] : 0;

  a = "anchors" in options ? options["anchors"] : 0; 
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
  return l;
}
function parse_yolo_mask(a) {
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
    return mask;
  }
  else return null;
}