"use strict";

class WebGLActivation {
  static createProgramInfo(handler, Layout, input, outputShape) {
    const glsl = getGlsl(handler.glContext.version);
    const shaderSource = getGlActivation(input.activation, glsl)
    if (shaderSource == null) return
    return {
      hasMain: true,
      inputLayouts: [handler.getOrCreateTextureLayout(input.TextureID, Layout.shape)],
      outputLayout: handler.createTextureLayoutFromShape(outputShape),
      samplers: ['A'],
      shaderSource,
    };
  }
  static createRunData(handler, glProg, outTextureID) {
    return {
      inputTextureDatas: [handler.getOrCreateTextureData(outTextureIDs, glProg.inputLayouts[0])],
      outputTextureData: handler.createTextureDataFromLayout(glProg.outputLayout, 'float32', outTextureID),
      uniformData: {}
    }
  }
}
function getGlActivation(a, glsl) {
  switch (a) {
    case "LOGISTIC":
      return glslSigmoid(glsl)
    case "RELU":
      return glslRelu(glsl);
    case "LEAKY":
      return glslLeakyRelu(glsl)
    case "MISH":
      return glslMish(glsl)
    case "SWISH":
      return glslSwish(glsl)
    default: return null;
  }
}
function glslElu(glsl) {
  const alpha = Math.exp(0.1)
  return `void main() {
    float v = ${glsl.texture2D}(A, TexCoords).r;
    ${glsl.output} = vec4(v >= 0.0 ? v: (exp(v) - 1.0) * ${alpha}); /* float number format */
  }`
}
function glslLeakyRelu(glsl) {
  const alpha = 0.1
  return `
  void main() {
    float v = ${glsl.texture2D}(A, TexCoords).r;
    ${glsl.output} = vec4(v < 0.0 ? v * float(${alpha}) : v);
  }
  `;
}
function glslRelu(glsl) {
  return `
  void main() {
    float v = ${glsl.texture2D}(A, TexCoords).r;
    ${glsl.output} = vec4(v < 0.0 ? 0.0 : v);
  }
  `;
}
function glslSigmoid(glsl) {
  return `
  void main() {
    float v = ${glsl.texture2D}(A, TexCoords).r;
    ${glsl.output} = vec4(1.0 / (1.0 + exp(-v)));
  }
  `;
}
function glslSwish(glsl) {
  return `
  void main() {
    float v = ${glsl.texture2D}(A, TexCoords).r;
    ${glsl.output} = vec4(v*(1.0 / (1.0 + exp(-v)));
  }
  `;
}
function glslTanh(glsl) {
  return `
  void main() {
    float v = ${glsl.texture2D}(A, TexCoords).r;
    v = clamp(v, -10., 10.);
    v = exp(2.*v);
    ${glsl.output} = vec4((v - 1.) / (v + 1.));
  }
  `;
}
function glslSoftplus(glsl) {
  const threshold = 20
  return `
  void main() {
    float threshold = float(${threshold});
    float v = ${glsl.texture2D}(A, TexCoords).r;
    if (v > threshold) ${glsl.output} = vec4(v);           
    else if (v < -threshold) ${glsl.output} = vec4(exp(v)); 
    else ${glsl.output} = vec4(log(exp(v) + 1.0));
  }`
}

function glslMish(glsl) {
  const threshold = 20
  return `
  float softplus(float v){
    float threshold = float(${threshold});
    if (v > threshold) v = v;           
    else if (v < -threshold) v = exp(v); 
    else v = log(exp(v) + 1.0);
    return v;
  }
  float tang(float v){
    v = clamp(v, -10., 10.);
    v = exp(2.*v);
    return (v - 1.) / (v + 1.);
  }
  void main() {
    float v = ${glsl.texture2D}(A, TexCoords).r;
    ${glsl.output} = vec4(v * tang(softplus(v)));
  }
  `
}