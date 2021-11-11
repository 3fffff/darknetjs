"use strict";
class WebGLSum {
  constructor(handler, l) {
    const glsl = getGlsl(handler.glContext.version);
    const outputShape = [l.b, l.c, l.h, l.w];
    const sumLine = l.output.map((v, i) => `${glsl.texture2D}(X${i},TexCoords)`).join(' + ');
    const samplers = l.output.map((v, i) => `X${i}`);
    const programInfo = {
      inputLayouts: l.output.map(t => handler.getOrCreateTextureLayout(t)),
      outputLayout: handler.createTextureLayoutFromShape(outputShape),
      samplers,
      shaderSource: `
      void main() {
        vec4 result = ${sumLine};
        ${glsl.output} = result;
      }`,
      hasMain: true
    };
    const inputTDs = inputs.map((t, i) => handler.getOrCreateTextureData(t, programInfo.inputLayouts[i]));
    return {
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].tensor.type),
      uniformData: {}
    };
  }
}