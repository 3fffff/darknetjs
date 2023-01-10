export function conv(l) {
  l.numThreads = (l.batch !== 1 || l.filters === 1 || WasmBinding.workerNumber <= 0) ? 1 : Math.min(l.filters, numWebWorkers + 1);
  if (l.numThreads > 1) {
    l.workerTasks = new Array(l.numThreads - 1);
    // data pre-processing
    l.wDimsSp = [l.filters, l.c / l.groups, l.size, l.size];
    l.wDimsSp[0] = l.groups == 1 ? ~~(l.filters / l.numThreads) : ~~(l.groups / l.numThreads);
    l.wSizeSp = l.wDimsSp[0] * l.wDimsSp[1] * l.wDimsSp[2] * l.wDimsSp[3];
    l.wDimsFinal = [l.filters, l.c / l.groups, l.size, l.size];
    l.wDimsFinal[0] = l.filters - (l.numThreads - 1) * l.wDimsSp[0];
    l.yDimsSp = [l.batch, l.wDimsSp[0], l.out_h, l.out_w];
    l.ySizeSp = l.wDimsSp[0] * l.out_h * l.out_w;
    l.yDimsFinal = [l.batch, l.wDimsFinal[0], l.out_h, l.out_w];
    l.xSizeSp = l.wDimsSp[0] * l.h * l.w;
    l.wArray = new Array(l.numThreads);
    l.yArray = new Array(l.numThreads);
    l.bArray = new Array(l.numThreads);
    l.sArray = new Array(l.numThreads);
    l.mArray = new Array(l.numThreads);
    l.vArray = new Array(l.numThreads);
    // function calls
    for (let i = 0; i < l.numThreads; ++i) {
      if (i !== l.numThreads - 1) {
        l.wArray[i] = l.weights.subarray(i * l.wSizeSp, (i + 1) * l.wSizeSp);
        l.yArray[i] = l.output.subarray(i * l.ySizeSp, (i + 1) * l.ySizeSp);
        if (l.biases) l.bArray[i] = l.biases.subarray(i * l.wDimsSp[0], (i + 1) * l.wDimsSp[0]);
        if (l.batch_normalize == 1) {
          l.sArray[i] = l.scales.subarray(i * l.wDimsSp[0], (i + 1) * l.wDimsSp[0]);
          l.mArray[i] = l.mean.subarray(i * l.wDimsSp[0], (i + 1) * l.wDimsSp[0]);
          l.vArray[i] = l.variance.subarray(i * l.wDimsSp[0], (i + 1) * l.wDimsSp[0]);
        }
      }
      else {
        l.wArray[i] = l.weights.subarray(i * l.wSizeSp);
        l.yArray[i] = l.output.subarray(i * l.ySizeSp);
        if (l.biases) l.bArray[i] = l.biases.subarray(i * l.wDimsSp[0]);
        if (l.batch_normalize == 1) {
          l.sArray[i] = l.scales.subarray(i * l.wDimsSp[0]);
          l.mArray[i] = l.mean.subarray(i * l.wDimsSp[0]);
          l.vArray[i] = l.variance.subarray(i * l.wDimsSp[0]);
        }
      }
    }
  }
}

export async function WasmConv(layers) {
  const l = this
  const X = layers[l.index - 1].output
  const active = { 'LOGISTIC': 1, 'RELU': 2, 'LEAKY': 3, 'LINEAR': 0, 'MISH': 4, 'SWISH': 5 }
  if (l.numThreads == 1)
    WasmBinding.getInstance().ccall(
      '_conv_f32', [X, 'float32ptr'], [[l.batch, l.c, l.h, l.w], 'int32ptr'], [l.weights, 'float32ptr'],
      [[l.filters, l.c, l.size, l.size], 'int32ptr'], [l.output, 'float32ptr', 'out'], [[l.batch, l.out_c, l.out_h, l.out_w], 'int32ptr'],
      [l.biases.length > 0 ? l.biases : null, 'float32ptr'], [[l.dilation, l.dilation], 'int32ptr'], [l.groups, 'int32'],
      [[l.pad, l.pad, l.pad, l.pad], 'int32ptr'], [[l.stride_x, l.stride_y], 'int32ptr'], [active[l.activation], 'int32'],
      [l.scales, 'float32ptr'], [l.mean, 'float32ptr'], [l.variance, 'float32ptr']);
  else {
    const workerTasks = new Array(l.numThreads - 1);
    // data pre-processing
    for (let i = 0; i < l.numThreads; ++i) {
      const xArray = new Array(l.numThreads);
      if (l.groups !== 1) xArray[i] = X.subarray(i * l.xSizeSp, (i + 1) * l.xSizeSp);
      if (i !== l.numThreads - 1) {
        workerTasks[i] = WasmBinding.getInstance().ccallRemote(i, '_conv_f32', [l.groups == 1 ? X : xArray[i], 'float32ptr'], [[l.batch, l.groups == 1 ? l.c : l.wDimsSp[0], l.h, l.w], 'int32ptr'],
          [l.wArray[i], 'float32ptr'], [l.wDimsSp, 'int32ptr'], [l.yArray[i], 'float32ptr', 'out'], [l.yDimsSp, 'int32ptr'],
          [l.biases ? l.bArray[i] : null, 'float32ptr'], [[l.dilation, l.dilation], 'int32ptr'], [l.groups == 1 ? l.groups : l.wDimsSp[0], 'int32'],
          [[l.pad, l.pad, l.pad, l.pad], 'int32ptr'], [[l.stride_x, l.stride_y], 'int32ptr'], [active[l.activation], 'int32'],
          [l.scales ? l.sArray[i] : null, 'float32ptr'], [l.mean ? l.mArray[i] : null, 'float32ptr'], [l.variance ? l.vArray[i] : null, 'float32ptr']);
      }
      else {
        WasmBinding.getInstance().ccall('_conv_f32', [l.groups == 1 ? X : xArray[i], 'float32ptr'], [[l.batch, l.groups == 1 ? l.c : l.wDimsSp[0], l.h, l.w], 'int32ptr'],
          [l.wArray[i], 'float32ptr'], [l.wDimsFinal, 'int32ptr'], [l.yArray[i], 'float32ptr', 'out'],
          [l.yDimsFinal, 'int32ptr'], [l.biases ? l.bArray[i] : null, 'float32ptr'],
          [[l.dilation, l.dilation], 'int32ptr'], [l.groups == 1 ? l.groups : l.wDimsSp[0], 'int32'], [[l.pad, l.pad, l.pad, l.pad], 'int32ptr'],
          [[l.stride_x, l.stride_y], 'int32ptr'], [active[l.activation], 'int32'],
          [l.scales ? l.sArray[i] : null, 'float32ptr'], [l.mean ? l.mArray[i] : null, 'float32ptr'], [l.variance ? l.vArray[i] : null, 'float32ptr']);
      }
    }
    await Promise.all(workerTasks);
  }
}