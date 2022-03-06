class WebGLSum {
  static createProgramInfo(handler, inputs, outputShape, outTextureID) {
    const glsl = getGlsl(handler.glContext.version);
    const sumLine = inputs.map((v, i) => `${glsl.texture2D}(X${i},TexCoords)`).join(getSamOrShortcut(inputs.type));
    const samplers = inputs.map((v, i) => `X${i}`);
    return {
      inputLayouts: inputs.map(t => handler.getOrCreateTextureLayout(t.TextureID, t.shape)),
      outputLayout: handler.createTextureLayoutFromShape(outputShape, 'float32', outTextureID),
      samplers,
      shaderSource: `
      void main() {
        vec4 result = ${sumLine};
        ${glsl.output} = result;
      }`,
      hasMain: true
    };
  }
  static createRunData(handler) {
    const inputTDs = this.textures.map((t, i) => handler.getOrCreateTextureData(t.TextureID, this.glProg.inputLayouts[i]));
    return {
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(this.glProg.outputLayout, 'float32', "t" + this.index),
      uniformData: {}
    };
  }
}
function getSamOrShortcut(type) {
  if (type == "SAM") return ' * '
  if (type == "SHORTCUT") return ' + '
  throw new Error("No recognized")
}