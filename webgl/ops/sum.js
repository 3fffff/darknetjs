class WebGLSum {
  static createProgramInfo(handler, inputs, outputShape, type) {
    const glsl = getGlsl(handler.glContext.version);
    const sumLine = inputs.map((v, i) => `${glsl.texture2D}(X${i},TexCoords)`).join(getSamOrShortcut(type));
    const samplers = inputs.map((v, i) => `X${i}`);
    return {
      inputLayouts: inputs.map(t => handler.getOrCreateTextureLayout(t.TextureID, t.shape)),
      outputLayout: handler.createTextureLayoutFromShape(outputShape),
      samplers,
      shaderSource: `
      void main() {
        vec4 result = ${sumLine};
        ${glsl.output} = result;
      }`,
      hasMain: true
    };
  }
  static createScaleChannelsProgramInfo(handler, inputs, outputShape) {
    const glsl = getGlsl(handler.glContext.version);
    const inputLayouts = inputs.map(t => handler.getOrCreateTextureLayout(t.TextureID, t.shape))
    const X = inputLayouts[1];
    const rank = inputLayouts[1].shape.length
    const shaderSource = `
    float process(int[${rank}] indices) {
      vec2 position = offsetToCoords(indices[1], ${X.width}, ${X.height});
      float c = getColorAsFloat(${glsl.texture2D}(C, position));

      return _X(indices) * c;
    }`;
    return {
      inputLayouts,
      outputLayout: handler.createTextureLayoutFromShape(outputShape),
      shaderSource,
      samplers: ['C', 'X'],
    };
  }
  static createRunData(handler, textures, glProg, outTextureID) {
    const inputTDs = textures.map((t, i) => handler.getOrCreateTextureData(t, glProg.inputLayouts[i]));
    return [{
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(glProg.outputLayout, 'float32', "t" + outTextureID),
      uniformData: {}
    }];
  }
}
function getSamOrShortcut(type) {
  if (type == "SAM") return ' * '
  if (type == "SHORTCUT") return ' + '
  throw new Error("No recognized")
}