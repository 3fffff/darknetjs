"use strict";
class WebGLConv {
  static createProgramInfos(handler, inputs, outputShape) {
    const im2colProgramInfo = createIm2ColProgramInfo(handler, inputs, outputShape);
    const dotProductProgramInfo = createDotProductProgramInfo(handler, im2colProgramInfo.outputLayout, inputs, outputShape);
    const activateProgramInfo = WebGLActivation.createProgramInfo(handler, dotProductProgramInfo.outputLayout, inputs[0], outputShape)
    return [im2colProgramInfo, dotProductProgramInfo, activateProgramInfo];
  }
  static createRunDatas(handler) {
    const b = this.textures.length >= 3 ? this.textures[2] : undefined;
    let kTD = handler.getTextureData(this.textures[1].TextureID);
    if (!kTD) {
      console.log('Conv', 'Did not find the adjustedKernel texture in the cache. Creating new.');
      const newKernelData = WebGLConv.prepKernelForDotProduct(this.textures[1].shape, this.textures[1].groups, 4, this.textures[1].weights);
      // hack: should use graph transformer to rewrite initializer K
      kTD = handler.createTextureDataFromLayoutBindTensor(this.glProg[1].inputLayouts[1], 'float32', newKernelData, this.textures[1]);
    }
    const runtDataIm2Col = {
      inputTextureDatas: [handler.getOrCreateTextureData(this.textures[0], null, this.textures[0].shape)],
      outputTextureData: handler.createTextureDataFromLayout(this.glProg[0].outputLayout, 'float32', "im2col" + this.index),
      uniformData: {}
    };
    const inputTDs = [runtDataIm2Col.outputTextureData, kTD];
    if (b) inputTDs.push(handler.getOrCreateTextureData(b, null, b.shape));
    const outputTD = handler.createTextureDataFromLayout(this.glProg[1].outputLayout, 'float32', this.activation == "LINEAR" ? "t" + this.index : "dotprod" + this.index);
    const runDataDotProduct = {
      inputTextureDatas: inputTDs,
      outputTextureData: outputTD,
      uniformData: {},
      draw: (glContext, artifact) => {
        const gl = glContext.gl;
        const sharedDim = artifact.programInfo.params.sharedDim;
        const sharedDimReadSize = artifact.programInfo.params.sharedDimReadSize;
        const sharedDimOffsetLocation = artifact.uniformLocations.find(l => l.name === 'sharedDimOffset').location;
        let blend = false;
        for (let k = 0; k < sharedDim; k += sharedDimReadSize) {
          //console.log('MatMul2D', `k = ${k}, sharedDim: ${sharedDim}, readSize = ${sharedDimReadSize}`);
          if (k === sharedDimReadSize) {
            blend = true;
            gl.enable(gl.BLEND);
            glContext.checkError();
            gl.blendEquation(gl.FUNC_ADD);
            glContext.checkError();
            gl.blendFunc(gl.ONE, gl.ONE);
            glContext.checkError();
          }
          gl.uniform1i(sharedDimOffsetLocation, k);
          glContext.checkError();
          glContext.draw();
        }
        if (blend) {
          gl.disable(gl.BLEND);
          glContext.checkError();
        }
      }
    };
    const runDataActivation = this.activation == "LINEAR" ? null : {
      inputTextureDatas: [runDataDotProduct.outputTextureData],
      outputTextureData: handler.createTextureDataFromLayout(this.glProg[2].outputLayout, 'float32', "t" + this.index),
      uniformData: {}
    }
    return [runtDataIm2Col, runDataDotProduct, runDataActivation];
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
      vec4 v = vec4(0.0);
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
            v[i] = _X(x);
          }
        }
        ++p;
      }
      return v;
    }
    `;
  return {
    inputLayouts: [handler.createTextureLayoutFromShape(xshape)],
    outputLayout,
    samplers: ['X'],
    shaderSource,
  };
}
function createDotProductProgramInfo(handler, im2colLayout, inputs, outputShape) {
  const xshape = inputs[0].shape
  const kshape = inputs[1].shape
  const adjustedKernelShape = [kshape[0], Math.ceil((xshape[1] * kshape[2] * kshape[3]) / xshape.length)];
  const kLayout = handler.createTextureLayoutFromShape(adjustedKernelShape, xshape.length, [adjustedKernelShape[0], adjustedKernelShape[1] * xshape.length], { breakAxis: 1 });
  let bLayout;
  const rank = outputShape.length;
  const inputLayouts = [im2colLayout, kLayout];
  if (inputs.length === 3) {
    bLayout = handler.createTextureLayoutFromShape(inputs[2].shape);
    inputLayouts.push(bLayout);
  }
  const outputLayout = handler.createTextureLayoutFromShape(outputShape);
  const initValue = (inputs.length < 3) ? '0.0' : '_B(b)';
  const sharedDim = im2colLayout.shape[3];
  const blendEnabled = handler.glContext.isBlendSupported;
  const sharedDimReadSize = blendEnabled && handler.matmulMaxBatchSize ? WebGLConv.calcSharedDimReadSize(handler.matmulMaxBatchSize, sharedDim) : sharedDim;
  const samplers = ['Im2Col', 'K'];
  if (inputs.length === 3) samplers.push('B');
  const glsl = getGlsl(handler.glContext.version);
  const shaderSource = `
  float process(int indices[${rank}]) {
    int b[1];
    b[0] = indices[1];
    int im2col[${im2colLayout.shape.length}];
    im2col[0] = indices[0];
    im2col[1] = indices[2];
    im2col[2] = indices[3];
    int im2colOffset = im2col[0] * ${im2colLayout.strides[0]} + im2col[1] * ${im2colLayout.strides[1]} + im2col[2] * ${im2colLayout.strides[2]} + sharedDimOffset;
    int kernelOffset = indices[1] * ${kLayout.strides[0]} + sharedDimOffset;
    float sum = sharedDimOffset == 0 ? ${initValue} : 0.0;
    for (int i = 0; i < ${sharedDimReadSize}; ++i) {
      vec2 im2colCoords = offsetToCoords(im2colOffset, ${im2colLayout.width}, ${im2colLayout.height});
      vec2 kernelCoords = offsetToCoords(kernelOffset, ${kLayout.width}, ${kLayout.height});
      sum += dot(${glsl.texture2D}(Im2Col, im2colCoords), ${glsl.texture2D}(K, kernelCoords));
      ++im2colOffset;
      ++kernelOffset;
    }
    return sum;
  }`;
  return {
    inputLayouts: inputs.length === 3 ? [im2colLayout, kLayout, bLayout] : [im2colLayout, kLayout],
    outputLayout,
    shaderSource,
    samplers,
    variables: [{ name: 'sharedDimOffset', type: 'int' }],
    params: { 'sharedDim': sharedDim, 'sharedDimReadSize': sharedDimReadSize }
  };
}