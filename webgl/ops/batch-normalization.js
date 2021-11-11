"use strict";

class WebGLBatchNormalization {
  static createProgramInfo(handler, inputs) {
    const inputLayouts = inputs.map(t => handler.getOrCreateTextureLayout(t));
    const outputShape = inputs[0].dims.slice();
    const rank = outputShape.length;
    const scale = inputLayouts[1];
    const glsl = getGlsl(handler.glContext.version);
    const shaderSource = `
      float process(int[${rank}] indices) {
        vec2 position = offsetToCoords(indices[1], ${scale.width}, ${scale.height});
        float scale = getColorAsFloat(${glsl.texture2D}(Scale, position));
        float mean = getColorAsFloat(${glsl.texture2D}(Mean, position));
        float variance = getColorAsFloat(${glsl.texture2D}(Variance, position));
        float b = getColorAsFloat(${glsl.texture2D}(B, position));

        return scale * ( (_A(indices) - mean) / sqrt(variance + float(${this.epsilon})) ) + b;
      }`;
    return {
      inputLayouts,
      outputLayout: handler.createTextureLayoutFromShape(outputShape),
      samplers: ['A', 'Scale', 'B', 'Mean', 'Variance'],
      shaderSource
    };
  }
  static createRunData(handler, programInfo, inputs) {
    const inputTDs = [handler.getOrCreateTextureData(inputs[0], programInfo.inputLayouts[0])];
    inputs.slice(1).forEach(t => inputTDs.push(handler.getOrCreateTextureData(t)));
    const outputTD = handler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].tensor.type);
    return { inputTextureDatas: inputTDs, outputTextureData: outputTD, uniformData: {} };
  }
}