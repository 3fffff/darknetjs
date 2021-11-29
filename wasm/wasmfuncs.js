async function WasmConv(layers) {
  const l = this
  const X = layers[l.index - 1].output
  const active = { 'LOGISTIC': 1, 'RELU': 2, 'LEAKY': 3, 'LINEAR': 0, 'MISH': 4, 'SWISH': 5 }
  const numThreads = (l.batch !== 1 || l.groups !== 1 || l.filters === 1 || WasmBinding.workerNumber <= 0) ? 1 : Math.min(l.filters, numWebWorkers + 1);
  if (numThreads == 1)
    WasmBinding.getInstance().ccall(
      '_conv_f32', [X, 'float32ptr'], [[l.batch, l.c, l.h, l.w], 'int32ptr'], [l.weights, 'float32ptr'],
      [[l.filters, l.c, l.size, l.size], 'int32ptr'], [l.output, 'float32ptr', 'out'], [[l.batch, l.out_c, l.out_h, l.out_w], 'int32ptr'],
      [l.biases.length > 0 ? l.biases : null, 'float32ptr'], [[l.dilation, l.dilation], 'int32ptr'], [l.groups, 'int32'],
      [[l.pad, l.pad, l.pad, l.pad], 'int32ptr'], [[l.stride_x, l.stride_y], 'int32ptr'], [active[l.activation], 'int32'],
      [l.scales, 'float32ptr'],[l.mean, 'float32ptr'],[l.variance, 'float32ptr']);
  else {
    const workerTasks = new Array(numThreads - 1);
    // data pre-processing
    const wDimsSp = [l.filters, l.c / l.groups, l.size, l.size];
    wDimsSp[0] = ~~(l.filters / numThreads);
    const wSizeSp = wDimsSp[0] * wDimsSp[1] * wDimsSp[2] * wDimsSp[3];
    const wDimsFinal = [l.filters, l.c / l.groups, l.size, l.size];
    wDimsFinal[0] = l.filters - (numThreads - 1) * wDimsSp[0];
    const yDimsSp = [l.batch, wDimsSp[0], l.out_h, l.out_w];
    const ySizeSp = wDimsSp[0] * l.out_h * l.out_w;
    const yDimsFinal = [l.batch, wDimsFinal[0], l.out_h, l.out_w];
    const wArray = new Array(numThreads);
    const yArray = new Array(numThreads);
    const bArray = new Array(numThreads);
    const sArray = new Array(numThreads);
    const mArray = new Array(numThreads);
    const vArray = new Array(numThreads);
    // function calls
    for (let i = 0; i < numThreads; ++i) {
      if (i !== numThreads - 1) {
        wArray[i] = l.weights.subarray(i * wSizeSp, (i + 1) * wSizeSp);
        yArray[i] = l.output.subarray(i * ySizeSp, (i + 1) * ySizeSp);
        if (l.biases) bArray[i] = l.biases.subarray(i * wDimsSp[0], (i + 1) * wDimsSp[0]);
        if (l.batch_normalize == 1){
          sArray[i] = l.scales.subarray(i * wDimsSp[0], (i + 1) * wDimsSp[0]);
          mArray[i] = l.mean.subarray(i * wDimsSp[0], (i + 1) * wDimsSp[0]);
          vArray[i] = l.variance.subarray(i * wDimsSp[0], (i + 1) * wDimsSp[0]);
        }
        workerTasks[i] = WasmBinding.getInstance().ccallRemote(i, '_conv_f32', [X, 'float32ptr'], [[l.batch, l.c, l.h, l.w], 'int32ptr'],
          [wArray[i], 'float32ptr'], [wDimsSp, 'int32ptr'], [yArray[i], 'float32ptr', 'out'], [yDimsSp, 'int32ptr'],
          [l.biases ? bArray[i] : null, 'float32ptr'], [[l.dilation, l.dilation], 'int32ptr'], [l.groups, 'int32'],
          [[l.pad, l.pad, l.pad, l.pad], 'int32ptr'], [[l.stride_x, l.stride_y], 'int32ptr'], [active[l.activation], 'int32'],
          [l.scales ? sArray[i] : null, 'float32ptr'],[l.mean ? mArray[i] : null, 'float32ptr'],[l.variance ? vArray[i] : null, 'float32ptr']);
      }
      else {
        wArray[i] = l.weights.subarray(i * wSizeSp);
        yArray[i] = l.output.subarray(i * ySizeSp);
        if (l.biases) bArray[i] = l.biases.subarray(i * wDimsSp[0]);
        if (l.batch_normalize == 1){
          sArray[i] = l.scales.subarray(i * wDimsSp[0]);
          mArray[i] = l.mean.subarray(i * wDimsSp[0]);
          vArray[i] = l.variance.subarray(i * wDimsSp[0]);
        }
        WasmBinding.getInstance().ccall('_conv_f32', [X, 'float32ptr'], [[l.batch, l.c, l.h, l.w], 'int32ptr'],
          [wArray[i], 'float32ptr'], [wDimsFinal, 'int32ptr'], [yArray[i], 'float32ptr', 'out'],
          [yDimsFinal, 'int32ptr'], [l.biases ? bArray[i] : null, 'float32ptr'],
          [[l.dilation, l.dilation], 'int32ptr'], [l.groups, 'int32'], [[l.pad, l.pad, l.pad, l.pad], 'int32ptr'],
          [[l.stride_x, l.stride_y], 'int32ptr'], [active[l.activation], 'int32'],
          [l.scales ? sArray[i] : null, 'float32ptr'],[l.mean ? mArray[i] : null, 'float32ptr'],[l.variance ? vArray[i] : null, 'float32ptr']);
      }
    }
    await Promise.all(workerTasks);
  }
}

async function WasmPool(layers) {
  const l = this
  const X = layers[l.index - 1].output
  const numThreads = (l.batch !== 1 || l.c === 1 || numWebWorkers <= 0) ? 1 : Math.min(l.c, numWebWorkers + 1);
  if (numThreads === 1)
    WasmBinding.getInstance().ccall('_pool_f32', [X, 'float32ptr'], [[l.batch, l.c, l.h, l.w], 'int32ptr'], [l.output, 'float32ptr', 'out'],
      [[l.batch, l.out_c, l.out_h, l.out_w], 'int32ptr'], [l.size, 'int32'], [[l.pad, l.pad], 'int32ptr'], [[l.stride_x, l.stride_y], 'int32ptr'],
      [l.type === 'AVGPOOL' ? true : false, 'bool']);
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
        [[l.pad, l.pad], 'int32ptr'], [[l.stride_x, l.stride_y], 'int32ptr'], [l.type === 'AVGPOOL' ? true : false, 'bool']);
      else WasmBinding.getInstance().ccall('_pool_f32', [X.subarray((numThreads - 1) * xSizeSp), 'float32ptr'], [xDimsFinal, 'int32ptr'],
        [l.output.subarray((numThreads - 1) * ySizeSp), 'float32ptr', 'out'], [yDimsFinal, 'int32ptr'], [l.size, 'int32'], [[l.pad, l.pad], 'int32ptr'],
        [[l.stride_x, l.stride_y], 'int32ptr'], [l.type === 'AVGPOOL' ? true : false, 'bool']);
    }
    await Promise.all(workerTasks);
  }
}