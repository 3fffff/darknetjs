import { getGlsl } from "../libglsl/glsl-source.js"

export function upsample(webgl, l) {
  const textures = [{ TextureID: "t" + (l.index - 1), scale: 1, stride: l.stride, shape: [l.batch, l.c, l.h, l.w] }]
  const glProg = createProgramInfo(webgl, textures[0], [l.batch, l.out_c, l.out_h, l.out_w])
  l.artifacts = [webgl.programManager.build(glProg)];
  l.runData = createRunData(webgl, textures[0], glProg, l.index)
  l.textures = textures
}
function createProgramInfo(handler, input, outputShape) {
  const inputLayout = handler.getOrCreateTextureLayout(input.TextureID, input.shape);
  const outputLayout = handler.createTextureLayoutFromShape(outputShape);
  const dim = outputShape.length
  const precalculatedPitches = shaderPrecalculatedPitches(dim, outputShape, input.shape);
  const glsl = getGlsl(handler.glContext.version)
  const shaderSource = `
    float process(int indices[${dim}]) {
      int output_index = coordsToOffset(TexCoords, ${outputLayout.width}, ${outputLayout.height});
      ${precalculatedPitches}
      int imageid = output_index / output_pitches[${dim - 3}]; 
      int m = output_index - imageid * output_pitches[${dim - 3}];
      int h = m / output_pitches[${dim - 2}];
      int w = m - h * output_pitches[${dim - 2}];
      int input_index = imageid * input_pitches[${dim - 3}] + int(h/${input.stride}) * input_pitches[${dim - 2}] + int(w/${input.stride});
      vec2 input_coord = offsetToCoords(input_index, ${inputLayout.width}, ${inputLayout.height});
      return float(${input.scale}) * getColorAsFloat(${glsl.texture2D}(X, input_coord));
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