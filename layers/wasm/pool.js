async function WasmPool(layers) {
    const l = this
    const X = layers[l.index - 1].output
    const type = {"MAXPOOL":1,"LOCALAVG":2,"AVG":3}
    const numThreads = (l.batch !== 1 || l.c === 1 || numWebWorkers <= 0) ? 1 : Math.min(l.c, numWebWorkers + 1);
    if (numThreads === 1)
      WasmBinding.getInstance().ccall('_pool_f32', [X, 'float32ptr'], [[l.batch, l.c, l.h, l.w], 'int32ptr'], [l.output, 'float32ptr', 'out'],
        [[l.batch, l.out_c, l.out_h, l.out_w], 'int32ptr'], [l.size, 'int32'], [[l.pad, l.pad], 'int32ptr'], [[l.stride_x, l.stride_y], 'int32ptr'],
        [type[l.type], 'bool']);
    else {
      // data pre-processing
      const xDimsSp = [l.batch, l.c, l.h, l.w];
      xDimsSp[1] = ~~(l.c / numThreads);
      const xSizeSp = xDimsSp[0] * xDimsSp[1] * xDimsSp[2] * xDimsSp[3];
      const xDimsFinal = [l.batch, l.c, l.h, l.w];
      xDimsFinal[1] = l.c - (numThreads - 1) * xDimsSp[1];
      const yDimsSp = [l.batch, xDimsSp[1], l.out_h, l.out_w];
      const ySizeSp = l.batch * xDimsSp[1] * l.out_h * l.out_w;
      const yDimsFinal = [l.batch, xDimsFinal[1], l.out_h, l.out_w];
      const workerTasks = new Array(numThreads - 1);
      // function calls
      for (let i = 0; i < numThreads; ++i) {
        if (i !== numThreads - 1) workerTasks[i] = WasmBinding.getInstance().ccallRemote(i, '_pool_f32', [X.subarray(i * xSizeSp, (i + 1) * xSizeSp), 'float32ptr'],
          [xDimsSp, 'int32ptr'], [l.output.subarray(i * ySizeSp, (i + 1) * ySizeSp), 'float32ptr', 'out'], [yDimsSp, 'int32ptr'], [l.size, 'int32'],
          [[l.pad, l.pad], 'int32ptr'], [[l.stride_x, l.stride_y], 'int32ptr'], [type[l.type], 'bool']);
        else WasmBinding.getInstance().ccall('_pool_f32', [X.subarray((numThreads - 1) * xSizeSp), 'float32ptr'], [xDimsFinal, 'int32ptr'],
          [l.output.subarray((numThreads - 1) * ySizeSp), 'float32ptr', 'out'], [yDimsFinal, 'int32ptr'], [l.size, 'int32'], [[l.pad, l.pad], 'int32ptr'],
          [[l.stride_x, l.stride_y], 'int32ptr'], [type[l.type], 'int32']);
      }
      await Promise.all(workerTasks);
    }
  }