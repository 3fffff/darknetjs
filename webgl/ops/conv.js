"use strict";
class WebGLConv {
  static createProgramInfos(handler, inputs, outputShape, activation, batch_normalize) {
    const im2colProgramInfo = createIm2ColProgramInfo(handler, inputs, outputShape);
    const dotProductProgramInfo = createDotProductProgramInfo(handler, im2colProgramInfo.outputLayout, inputs, outputShape, activation);
    const BNProgramInfo = batch_normalize ? WebGLBatchNormalization.createProgramInfo(handler, dotProductProgramInfo.outputLayout, inputs, outputShape, activation) : null;
    return [im2colProgramInfo, dotProductProgramInfo].filter(x => !!x);
  }
  static createRunDatas(handler, textures, glProg, outTextureID, batch_normalize) {
    const b = textures.length >= 3 ? textures[2] : undefined;
    let kTD = handler.getTextureData(textures[1].TextureID);
    if (!kTD) {
      console.log('Conv', 'Did not find the adjustedKernel texture in the cache. Creating new.');
      const newKernelData = WebGLConv.prepKernelForDotProduct(textures[1].shape, textures[1].groups, 4, textures[1].weights);
      // hack: should use graph transformer to rewrite initializer K
      kTD = handler.createTextureDataFromLayoutBindTensor(glProg[1].inputLayouts[1], 'float32', newKernelData, textures[1]);
    }
    const runtDataIm2Col = {
      inputTextureDatas: [handler.getOrCreateTextureData(textures[0], null, textures[0].shape)],
      outputTextureData: handler.createTextureDataFromLayout(glProg[0].outputLayout, 'float32', "im2col" + outTextureID),
      uniformData: {}
    };
    const inputTDs = [runtDataIm2Col.outputTextureData, kTD];
    if (b) inputTDs.push(handler.getOrCreateTextureData(b, null, b.shape));
    const outputTD = handler.createTextureDataFromLayout(glProg[1].outputLayout, 'float32', "t" + outTextureID);
    const runDataDotProduct = {
      inputTextureDatas: inputTDs,
      outputTextureData: outputTD,
      uniformData: {},
    };
    const inputATDs = [runtDataIm2Col.outputTextureData, kTD];
    const runDataBN = batch_normalize ? {
      inputTextureDatas: inputATDs,
      outputTextureData: outputTD,
      uniformData: {},
    } : null;
    return [runtDataIm2Col, runDataDotProduct].filter(x => !!x);
  }
  static prepKernelForDotProduct(shape, group, channels, kernel) {
    if (group === 1 && (channels === 1 || (shape[2] * shape[3]) % channels === 0)) return kernel;
    const numFeatureMaps = shape[0];
    const oldRowSize = shape[1] * shape[2] * shape[3];
    const newRowSize = Math.ceil(oldRowSize * group / channels) * channels;
    const newSize = numFeatureMaps * newRowSize;
    const buffer = new Float32Array(newSize);
    for (let f = 0; f < numFeatureMaps; ++f) {
      const oldOffset = f * oldRowSize;
      const newOffset = f * newRowSize + f % group * oldRowSize;
      buffer.set(kernel.subarray(oldOffset, oldOffset + oldRowSize), newOffset);
    }
    return buffer;
  }
  static calcIm2ColDims(inputShape, kernelShape, outputShape, channels = 1) {
    return [
      outputShape[0], outputShape[2], outputShape[3],
      Math.ceil(inputShape[1] * kernelShape[2] * kernelShape[3] / channels)
    ];
  }
  static calcSharedDimReadSize(preferredBatchSize, sharedDim) {
    if (preferredBatchSize <= 0 || sharedDim < preferredBatchSize || sharedDim % preferredBatchSize !== 0)
      return sharedDim;
    return preferredBatchSize;
  }
}
function createIm2ColProgramInfo(handler, inputs, outputShape) {
  const xshape = inputs[0].shape;
  const kshape = inputs[1].shape
  const rank = outputShape.length;
  const im2colDims = WebGLConv.calcIm2ColDims(xshape, kshape, outputShape, rank);
  const outputLayout = handler.createTextureLayoutFromShape(im2colDims, rank, [im2colDims[0], im2colDims[1], im2colDims[2], im2colDims[3] * rank], { breakAxis: 3 });
  const shaderSource = `
    const int XC = ${xshape[1]};
    const int XH = ${xshape[2]};
    const int XW = ${xshape[3]};
    const int KH = ${inputs[1].size};
    const int KW = ${inputs[1].size};
    const int dilationH = ${inputs[1].dilation};
    const int dilationW = ${inputs[1].dilation};
    const int strideH = ${inputs[1].stride_x};
    const int strideW = ${inputs[1].stride_y};
    const int padH = ${inputs[1].pad};
    const int padW = ${inputs[1].pad};
    const int KHKW = KH*KW;
    const int XCKHKW = XC * KHKW;
    const int outputChannels = ${xshape.length};

    vec4 process(int indices[${rank}]) {
      int b  = indices[0]; // batch size
      int oh = indices[1] * strideH - padH; //output height
      int ow = indices[2] * strideW - padW; //output width
      int p = indices[3] * outputChannels; //patch
      vec4 value = vec4(0.0);
      for(int i=0; i < outputChannels; ++i) {
        if(p < XCKHKW) {
          int patchC = p / KHKW;
          int patchH = (p - patchC*KHKW) / KW;
          int patchW = (p - patchC*KHKW) - patchH * KW;
          int xh2 = oh + patchH * dilationH;
          int xw2 = ow + patchW * dilationW;
          int x[${xshape.length}];
          x[0] = b;
          x[1] = patchC;
          x[2] = xh2;
          x[3] = xw2;
          if(xh2 >= 0 &&
              xh2 < XH &&
              xw2 >= 0 &&
              xw2 < XW) {
            value[i] = _X(x);
          }
        }
        ++p;
      }
      return value;
    }
    `;
  return {
    inputLayouts: [handler.createTextureLayoutFromShape(xshape)],
    outputLayout,
    samplers: ['X'],
    shaderSource,
  };
}
function createDotProductProgramInfo(handler, im2colLayout, inputs, outputShape, activ) {
  const { funcActivation, nameActivation } = activ == "LINEAR" ? { funcActivation: ``, nameActivation: `` } : getGlActivation(activ)
  const xshape = inputs[0].shape
  const kshape = inputs[1].shape
  const adjustedKernelShape = [kshape[0], Math.ceil((xshape[1] * kshape[2] * kshape[3]) / (xshape.length * inputs[1].groups))];
  const kLayout = handler.createTextureLayoutFromShape(adjustedKernelShape, xshape.length, [adjustedKernelShape[0], adjustedKernelShape[1] * xshape.length * inputs[1].groups], { breakAxis: 1 });
  const rank = outputShape.length;
  const inputLayouts = [im2colLayout, kLayout];
  let bLayout;
  if (inputs.length === 3) {
    bLayout = handler.createTextureLayoutFromShape(inputs[2].shape);
    inputLayouts.push(bLayout);
  }
  const outputLayout = handler.createTextureLayoutFromShape(outputShape);
  const initValue = (inputs.length < 3) ? '0.0' : '_B(b)';
  const sharedDim = Math.ceil(xshape[1] * kshape[2] * kshape[3] / (4 * inputs[1].groups * inputs[1].groups));
  const samplers = ['Im2Col', 'K'];
  if (inputs.length === 3) samplers.push('B');
  const glsl = getGlsl(handler.glContext.version);
  const shaderSource = `
  ${funcActivation}
  float process(int indices[${rank}]) {
    int b[1];
    b[0] = indices[1];
    int im2col[4];
    im2col[0] = indices[0];
    im2col[1] = indices[2];
    im2col[2] = indices[3];
    int im2colOffset = im2col[0] * ${im2colLayout.strides[0]} + im2col[1] * ${im2colLayout.strides[1]} + im2col[2] * ${im2colLayout.strides[2]} ;
    int kernelOffset = indices[1] * ${adjustedKernelShape[1]};
    float value = ${initValue};
    for (int i = 0; i < ${sharedDim}; ++i) {
      vec2 im2colCoords = offsetToCoords(im2colOffset, ${im2colLayout.width}, ${im2colLayout.height});
      vec2 kernelCoords = offsetToCoords(kernelOffset, ${kLayout.width}, ${kLayout.height});
      value += dot(${glsl.texture2D}(Im2Col, im2colCoords), ${glsl.texture2D}(K, kernelCoords));
      ++im2colOffset;
      ++kernelOffset;
    }
    ${nameActivation}
    return value;
  }`;
  return {
    inputLayouts: inputs.length === 3 ? [im2colLayout, kLayout, bLayout] : [im2colLayout, kLayout],
    outputLayout,
    shaderSource,
    samplers,
    params: { 'sharedDim': sharedDim }
  };
}

class WebGLGroupConv {
  static createProgramInfos(handler, inputs, outputShape, activation, batch_normalize) {
    const groupConvProgramInfo = createGroupConvProgramInfo(handler, inputs, outputShape, activation);
    const batchnormProgramInfo = batch_normalize ? WebGLBatchNormalization.createProgramInfo(handler, groupConvProgramInfo.outputLayout, inputs, outputShape, activation) : null;
    return [groupConvProgramInfo].filter(x => !!x)
  }
  static createRunDatas(handler, texture, glProg, outTextureID) {
    const inputTDs = texture.map((t, i) => handler.getOrCreateTextureData(t, glProg[0].inputLayouts[i]));
    return [{
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(glProg[0].outputLayout, 'float32', outTextureID),
      uniformData: {}
    }];
  }
}
function createGroupConvProgramInfo(handler, inputs, outputShape, activation, batch_normalize) {
  const { funcActivation, nameActivation } = activation == "LINEAR" ? { funcActivation: ``, nameActivation: `` } : getGlActivation(activation)
  const hasBias = inputs.length > 2;
  const processBias = hasBias ? `value += getBias(output_channel);` : ``;
  const xShape = inputs[0].shape;
  const wShape = inputs[1].shape;
  const outputChannelsPerGroup = wShape[0] / inputs[1].groups;
  const glsl = getGlsl(handler.glContext.version);
  const samplers = hasBias ? ['X', 'W', 'Bias'] : ['X', 'W']
  const shaderSource = `
  ${funcActivation}
  const ivec2 strides = ivec2(${inputs[1].stride_x}, ${inputs[1].stride_y});
  const ivec2 pads = ivec2(${inputs[1].pad}, ${inputs[1].pad});
  void main() {
    ivec4 coords = getOutputCoords();
    int batch = coords.x;
    int output_channel = coords.y;
    ivec2 xRCCorner = coords.zw * strides - pads;
    int group_id = output_channel / ${outputChannelsPerGroup};
    float value = 0.0;
    for (int wInChannel = 0; wInChannel < ${wShape[1]}; wInChannel++) {
      int input_channel = group_id * ${wShape[1]} + wInChannel;
      for (int wHeight = 0; wHeight < ${wShape[2]}; wHeight++) {
        int xHeight = xRCCorner.x + wHeight * ${inputs[1].dilation};
        if (xHeight < 0 || xHeight >= ${xShape[2]}) {
          continue;
        }
        for (int wWidth = 0; wWidth < ${wShape[3]}; wWidth++) {
          int xWidth = xRCCorner.y + wWidth * ${inputs[1].dilation};
          if (xWidth < 0 || xWidth >= ${xShape[3]}) {
            continue;
          }
          float xVal = getX(batch, input_channel, xWidth, xHeight);
          float wVal = getW(output_channel, wInChannel, wWidth, wHeight);
          value += xVal*wVal;
        }
      }
    }
    ${processBias}
    ${nameActivation}
    ${glsl.output} = vec4(value, .0, .0, .0);
  }
`;
  return {
    inputLayouts: inputs.map(t => handler.getOrCreateTextureLayout(t.TextureID, t.shape)),
    outputLayout: handler.createTextureLayoutFromShape(outputShape),
    samplers: samplers,
    shaderSource,
    hasMain: true,
  };
}