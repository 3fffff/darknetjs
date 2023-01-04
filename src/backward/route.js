export function route(layers) {
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