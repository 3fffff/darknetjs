"use strict";
class WebGLMatMul {
  static createProgramInfo(handler, inputs, outputShape) {
    const sharedDim = inputs[0].shape[inputs[0].length - 1];
    const line = `value += _A(a) * _B(b);`;
    const rank = outputShape.length;
    const shaderSource = `
      float process(int indices[${rank}]) {
          int a[${rank}];
          int b[${rank}];

          copyVec(indices, a);
          copyVec(indices, b);

          float value = 0.0;
          for (int k=0; k<${sharedDim}; ++k) {
              a[${rank - 1}] = k;
              b[${rank - 2}] = k;
              ${line}
          }
          return value;
      }`;
    const inputLayouts = inputs.map(t => handler.getOrCreateTextureLayout(t.TextureID, t.shape));
    return {
      inputLayouts,
      outputLayout: handler.createTextureLayoutFromShape(outputShape),
      samplers: ['A', 'B'],
      shaderSource,
    };
  }
  static createRunData(handler, textures, glProg, outTextureID) {
    const inputTDs = textures.map((t, i) => handler.getOrCreateTextureData(t, glProg.inputLayouts[i]));
    return {
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(glProg.outputLayout, "float32", "t" + outTextureID),
      uniformData: {}
    };
  }
}