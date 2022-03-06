"use strict";

class WebGLBatchNormalization {
  static createProgramInfo(handler, inputs, outputShape) {
    const inputLayouts = inputs.map(t => handler.getOrCreateTextureLayout(t.TextureID, t.shape));
    const rank = outputShape.length;
    const scale = inputLayouts[1];
    const glsl = getGlsl(handler.glContext.version);
    const epsilon = 0.000001
    const shaderSource = `
      float process(int[${rank}] indices) {
        vec2 position = offsetToCoords(indices[1], ${scale.width}, ${scale.height});
        float scale = getColorAsFloat(${glsl.texture2D}(Scale, position));
        float mean = getColorAsFloat(${glsl.texture2D}(Mean, position));
        float variance = getColorAsFloat(${glsl.texture2D}(Variance, position));
        float bias = getColorAsFloat(${glsl.texture2D}(Bias, position));

        return scale * ( (_A(indices) - mean) / sqrt(variance + float(${epsilon})) ) + bias;
      }`;
    return {
      inputLayouts,
      outputLayout: handler.createTextureLayoutFromShape(outputShape),
      samplers: ['A', 'Scale', 'Bias', 'Mean', 'Variance'],
      shaderSource
    };
  }
  static createRunData(handler) {
    const inputTDs = [handler.getOrCreateTextureData(inputs, this.glProg.inputLayouts)];
    inputs.slice(1).forEach(t => inputTDs.push(handler.getOrCreateTextureData(t)));
    console.log(inputTDs)
    const outputTD = handler.createTextureDataFromLayout(this.glProg.outputLayout, 'float32');
    return { inputTextureDatas: inputTDs, outputTextureData: outputTD, uniformData: {} };
  }
}