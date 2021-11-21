"use strict";
class WebGLUpsample {
  createProgramInfo(handler,input,outputShape) {
    const inputLayout = handler.getOrCreateTextureLayout(input.TextureID,input.shape);
    const outputLayout = handler.createTextureLayoutFromShape(outputShape);
    const glsl = getGlsl(handler.glContext.version);
    return createUpsampleProgramInfo(glsl, 4, inputLayout, outputLayout);
  }
  createRunData(handler,inputs) {
    const inputTD = handler.getOrCreateTextureData(inputs, this.glProg.inputLayouts);
    const outputTD = handler.createTextureDataFromLayout(this.glProg.outputLayout, "float32",this);
    return {
      inputTextureDatas: [inputTD],
      outputTextureData: outputTD,
      uniformData: {scales: [1,1,this.stride,this.stride]}
    };
  }
}
function createUpsampleProgramInfo(glsl, dim, inputLayout, outputLayout) {
  const outputShape = outputLayout.shape;
  const inputShape = inputLayout.shape;
  const precalculatedPitches = shaderPrecalculatedPitches(dim, outputShape, inputShape);
  const getInputFloatFunction = shaderGetInputFloatFunction(inputLayout, glsl);
  const shaderSource =
    // nearest
    `${getInputFloatFunction}
        float process(int indices[${dim}]) {
          int input_index = 0;
          int output_index = coordsToOffset(TexCoords, ${outputLayout.width}, ${outputLayout.height});
          ${precalculatedPitches}
          int d, m;
          for (int dim = 0; dim < ${dim}; ++dim) {
            d = output_index / output_pitches[dim];
            m = output_index - d * output_pitches[dim];
            output_index = m;
            if (scales[dim] != 1 && d > 0) {
              int d2 = d / scales[dim];
              m = d - d2 * scales[dim];
              d = d2;
            }
            input_index += input_pitches[dim] * d;
          }
          return getInputFloat(input_index);
        }`
  return {
    inputLayouts: [inputLayout],
    outputLayout,
    samplers: ['X'],
    shaderSource,
    variables: [{ name: 'scales', type: 'int', arrayLength: 4 }]
  };
}
function shaderPrecalculatedPitches(dim, outputShape, inputShape) {
  const outputPitches = new Array(dim);
  const inputPitches = new Array(dim);
  let precalculatedPitches = `
        int output_pitches[${dim}];
        int input_pitches[${dim}];
        `;
  for (let d = dim - 1; d >= 0; d--) {
    outputPitches[d] = (d === dim - 1) ? 1 : outputPitches[d + 1] * outputShape[d + 1];
    inputPitches[d] = (d === dim - 1) ? 1 : inputPitches[d + 1] * inputShape[d + 1];
    precalculatedPitches += `
        output_pitches[${d}] = ${outputPitches[d]};
        input_pitches[${d}] = ${inputPitches[d]};
        `;
  }
  return precalculatedPitches;
}
function shaderGetInputFloatFunction(inputLayout, glsl) {
  return `
float getInputFloat(int index) {
  vec2 coords = offsetToCoords(index, ${inputLayout.width}, ${inputLayout.height});
  float value = getColorAsFloat(${glsl.texture2D}(X, coords));
  return value;
}
`;
}