"use strict";

class WebGLActivation {
  static createProgramInfo(handler, l) {
    const outputShape = [l.b, l.out_c, l.out_h, l.out_w];
    this.alpha = 0.1
    const glsl = getGlsl(handler.glContext.version);
    const shaderSource = `
      void main() {
        float v = ${glsl.texture2D}(A, TexCoords).r;
        ${glsl.output} = vec4(v < 0.0 ? v * float(${this.alpha}) : v);
      }
      `;
    return {
      hasMain: true,
      inputLayouts: [handler.getOrCreateTextureLayout(l)],
      outputLayout: handler.createTextureLayoutFromShape(outputShape),
      samplers: ['A'],
      shaderSource,
    };
  }
  static createRunData(handler, l,input) {
    const inputTDs = [handler.getOrCreateTextureData(l,input, l.glProg.inputLayouts[0])];
    return {
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(l.glProg.outputLayout,'float32',l),
      uniformData: {}
    }
  }
}
function glslRelu() {
  return `
  void main() {
    float v = ${glsl.texture2D}(A, TexCoords).r;
    ${glsl.output} = vec4(v < 0.0 ? 0.0 : v);
  }
  `;
}
function glslSigmoid() {
  return`
  void main() {
    float v = ${glsl.texture2D}(A, TexCoords).r;
    ${glsl.output} = vec4(1.0 / (1.0 + exp(-v));
  }
  `;
}
function glslTanh() {
  return`
  void main() {
    float v = ${glsl.texture2D}(A, TexCoords).r;
    v = clamp(v, -10., 10.);
    v = exp(2.*v);
    ${glsl.output} = vec4((v - 1.) / (v + 1.));
  }
  `;
}