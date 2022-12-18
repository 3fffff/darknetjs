export function pool(webgl,l){
  const textures = [{ TextureID: "t" + (l.index - 1), pad: l.pad, size: l.size, stride_x: l.stride_x, stride_y: l.stride_x, shape: [l.batch, l.c, l.h, l.w] }]
  const glProg = createProgramInfo(webgl, textures[0], [l.batch, l.out_c, l.out_h, l.out_w], l.type)
  l.artifacts = [webgl.programManager.build(glProg)]
  l.runData = createRunData(webgl, textures, glProg, l.index)
}

function createProgramInfo(handler, inputs, outputShape, type) {
  if (type == 'MAXPOOL') return createMaxPoolProgramInfo(handler, inputs, outputShape, type);
  else return createAveragePoolProgramInfo(handler, inputs, outputShape, type);
}
function createRunData(handler, textures, glProg, outTextureID) {
  return [{
    inputTextureDatas: [handler.getOrCreateTextureData(textures[0], glProg.inputLayouts[0])],
    outputTextureData: handler.createTextureDataFromLayout(glProg.outputLayout, "float32", "t" + outTextureID),
    uniformData: {}
  }];
}
function createMaxPoolProgramInfo(handler, input, outputShape, type) {
  const op1 = `value = max(_X(x), value);`;
  const inputLayout = handler.createTextureLayoutFromShape(input.shape);
  const poolingCode = GeneratePoolingCode(inputLayout, [input.size, input.size], [input.pad, input.pad, input.pad, input.pad], [input.stride_x, input.stride_y], op1, '', '-1e5', type);
  const shaderSource = `${poolingCode}`;
  return {
    inputLayouts: [inputLayout],
    outputLayout: handler.createTextureLayoutFromShape(outputShape),
    samplers: ['X'],
    shaderSource,
  };
}
function createAveragePoolProgramInfo(handler, input, outputShape, type) {
  const kernelSize = getSizeFromDimensionRange([input.size, input.size], 0, 2);
  const op1 = `value += _X(x);`;
  const op2 = `value /= float(${kernelSize} - pad);`;
  const inputLayout = handler.getOrCreateTextureLayout(input, input.shape);
  const poolingCode = GeneratePoolingCode(inputLayout, [input.size, input.size], [input.pad, input.pad, input.pad, input.pad], [input.strides_x, input.strides_y], op1, op2, '0.0', type);
  const shaderSource = `
      ${poolingCode}
    `;
  return {
    inputLayouts: [inputLayout],
    outputLayout: handler.createTextureLayoutFromShape(outputShape),
    samplers: ['X'],
    shaderSource,
  };
}
function GeneratePoolingCode(x, kernelShape, pads, strides, op1, op2, startVal, type) {
  const inputDims = x.shape;
  const rank = x.shape.length;
  const kw = kernelShape[kernelShape.length - 1];
  const sw = strides[strides.length - 1];
  const pwStart = pads[pads.length / 2 - 1];
  const pwEnd = pads[pads.length - 1];
  const dimW = inputDims[rank - 1];
  let codeW = ``;
  let codeH = ``;
  let codeHEnd = ``;
  if (pwStart + pwEnd !== 0) {
    codeW = `
                for (int i = 0; i < ${kw}; i++) {
                  x[${rank} - 1] = indices[${rank} - 1] * ${sw} - ${pwStart} + i;
                  if (x[${rank} - 1] < 0 || x[${rank} - 1] >= ${dimW}) {
                    pad++;
                    continue;
                  }
                  ${op1}
                }`;
  }
  else {
    codeW = `
                for (int i = 0; i < ${kw}; i++) {
                  x[${rank} - 1] = indices[${rank} - 1] * ${sw} - ${pwStart} + i;
                  ${op1}
                }`;
  }
  const kh = kernelShape[kernelShape.length - 2];
  const sh = strides[strides.length - 2];
  const phStart = pads[pads.length / 2 - 2];
  const phEnd = pads[pads.length - 2];
  const dimH = inputDims[rank - 2];
  if (phStart + phEnd !== 0) {
    codeH = `
              for (int j = 0; j < ${kh}; j++) {
                x[${rank} - 2] = indices[${rank} - 2] * ${sh} - ${phStart} + j;
                if (x[${rank} - 2] < 0 || x[${rank} - 2] >= ${dimH}) {
                  pad+= ${kw};
                  continue;
                }
            `;
  }
  else {
    codeH = `
                for (int j = 0; j < ${kh}; j++) {
                  x[${rank} - 2] = indices[${rank} - 2] * ${sh} - ${phStart} + j;
                `;
  }
  codeHEnd = ` }`;
  let code = ``
  if (type == 'AVGPOOL') {
    code = `
  for (int j = 0; j < ${inputDims[1]}; j++) {
    x[${rank} - 2] = j * ${inputDims[2]} * ${inputDims[3]};
    ${op1}
  }`;
  } else code = codeH + codeW + codeHEnd + op2
  const poolingCode = `
            float process(int indices[${rank}]) {
              int x[${rank}];
              copyVec(indices, x);

              float value = ${startVal};
              int pad = 0;
              ${code}
              return value;
            }
          `;
  return poolingCode;
}

function getSizeFromDimensionRange(dims, start, end) {
  let size = 1;
  for (let i = start; i < end; i++) {
    // safety check as this method is called by multiple other methods requiring size.
    // size cannot be 0 or negative.
    if (dims[i] <= 0) {
      throw new Error(
        `cannot get valid size from specified dimension range. Most likely the range contains 0 or negative values in them.`);
    }
    size *= dims[i];
  }
  return size;
}