"use strict";
class WebGLUpsample {
  createProgramInfo(handler, inputs) {
    const inputLayout = handler.getOrCreateTextureLayout(inputs[0]);
    const [roi, scales, outputShape] = this.prepare(inputs);
    this.roiCache = roi;
    this.scalesCache = scales.map(x => Math.ceil(x));
    const outputLayout = handler.createTextureLayoutFromShape(outputShape);
    const dim = outputShape.length;
    const glsl = getGlsl(handler.session.backend.glContext.version);
    return createUpsampleProgramInfo(glsl, this.mode, dim, inputLayout, outputLayout, scales);
  }
  createRunData(handler, programInfo, inputs) {
    const inputTD = handler.getOrCreateTextureData(inputs[0], programInfo.inputLayouts[0]);
    const outputTD = handler.createTextureDataFromLayout(programInfo.outputLayout, inputTD.tensor.type);
    return {
      inputTextureDatas: [inputTD],
      outputTextureData: outputTD,
      uniformData: {
        scales: this.scalesCache,
        mo: this.mappingOriginCache,
        me: this.mappingExtrapolateCache,
        mw: this.mappingWeightCache,
        mc: this.mappingCoeffCache
      }
    };
  }
}
function fillResizeNearestMapping2D(inputHeight, inputWidth, outputHeight, outputWidth, scalesHeight, scalesWidth, roiStartHeight, roiEndHeight, roiStartWidth, roiEndWidth, extrapolationEnabled, getOriginalCoordinate, getNearestPixel, mappingOrigin, mappingExtrapolation) {
  for (let i = 0; i < outputHeight; i++) {
    let dim = i;
    const originalCoord = getOriginalCoordinate(dim, scalesHeight, outputHeight, inputHeight, roiStartHeight, roiEndHeight);
    // extrapolate
    mappingExtrapolation.push((extrapolationEnabled && (originalCoord < 0 || originalCoord > inputHeight - 1)) ? 1 : 0);
    dim = Math.max(0, Math.min(inputHeight - 1, getNearestPixel(originalCoord, scalesHeight < 1)));
    // origin
    mappingOrigin.push(dim);
  }
  for (let i = 0; i < outputWidth; i++) {
    let dim = i;
    const originalCoord = getOriginalCoordinate(dim, scalesWidth, outputWidth, inputWidth, roiStartWidth, roiEndWidth);
    // extrapolate
    mappingExtrapolation.push((extrapolationEnabled && (originalCoord < 0 || originalCoord > inputWidth - 1)) ? 1 : 0);
    dim = Math.max(0, Math.min(inputWidth - 1, getNearestPixel(originalCoord, scalesWidth < 1)));
    // origin
    mappingOrigin.push(dim);
  }
}
function createUpsampleProgramInfo(glsl, mode, dim, inputLayout, outputLayout, scales) {
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
    variables: [{ name: 'scales', type: 'int', arrayLength: scales.length }]
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