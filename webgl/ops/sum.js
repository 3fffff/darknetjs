class WebGLSum {
  static createProgramInfo(handler,inputs,outputShape) {
    const glsl = getGlsl(handler.glContext.version);
    const sumLine = inputs.map((v, i) => `${glsl.texture2D}(X${i},TexCoords)`).join(getSamOrShortcut(inputs.type));
    const samplers = inputs.map((v, i) => `X${i}`);
    return {
      inputLayouts: inputs.map(t => handler.getOrCreateTextureLayout(t.TextureID,t.shape)),
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
  static createRunData(handler,inputs,glProg,layer) {
    const inputTDs = inputs.map((t, i) => handler.getOrCreateTextureData(t.TextureID, glProg.inputLayouts[i]));
    console.log(inputTDs)
    return {
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(glProg.outputLayout, 'float32',layer),
      uniformData: {}
    };
  }
}
function getSamOrShortcut(type){
  if (type == "SAM")return ' * '
  if (type == "SHORTCUT")return ' + '
  throw new Error("No recognized")
}