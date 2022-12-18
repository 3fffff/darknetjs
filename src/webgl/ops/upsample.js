export function upsample(webgl,l){
  const textures = [{ TextureID: "t" + (l.index - 1), scale: 1, stride: l.stride, shape: [l.batch, l.c, l.h, l.w] }]
  const glProg = createProgramInfo(webgl, textures[0], [l.batch, l.out_c, l.out_h, l.out_w])
  l.artifacts = [webgl.programManager.build(glProg)];
  l.runData = createRunData(webgl, textures[0], glProg, l.index)
}
function createProgramInfo(handler, input, outputShape) {
  const inputLayout = handler.getOrCreateTextureLayout(input.TextureID, input.shape);
  const outputLayout = handler.createTextureLayoutFromShape(outputShape);
  const glsl = getGlsl(handler.glContext.version);
  const dim = outputShape.length
  const precalculatedPitches = shaderPrecalculatedPitches(dim, outputShape, input.shape);
  const getInputFloatFunction = shaderGetInputFloatFunction(inputLayout, glsl);
  const shaderSource = `
    ${getInputFloatFunction}
    float process(int indices[${dim}]) {
      int output_index = coordsToOffset(TexCoords, ${outputLayout.width}, ${outputLayout.height});
      ${precalculatedPitches}
      int imageid = output_index / output_pitches[${dim - 3}]; 
      int m = output_index - imageid * output_pitches[${dim - 3}];
      int h = m / output_pitches[${dim - 2}];
      int w = m - h * output_pitches[${dim - 2}];
      int input_index = imageid * input_pitches[${dim - 3}] + int(h/${input.stride}) * input_pitches[${dim - 2}] + int(w/${input.stride});
      return float(${input.scale}) * getInputFloat(input_index);
    }`
  return {
    inputLayouts: [inputLayout],
    outputLayout,
    samplers: ['X'],
    shaderSource
  };
}
function createRunData(handler, texture, glProg, outTextureID) {
  return [{
    inputTextureDatas: [handler.getOrCreateTextureData(texture, glProg.inputLayouts[0])],
    outputTextureData: handler.createTextureDataFromLayout(glProg.outputLayout, "float32", "t" + outTextureID)
  }];
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