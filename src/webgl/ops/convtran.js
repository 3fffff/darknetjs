class WebGLConvTranspose {
  static createProgramInfos(handler, inputs, outputShape, activation, batch_normalize) {
    const groupConvProgramInfo = createGroupConvProgramInfo(handler, inputs, outputShape, activation);
    const batchnormProgramInfo = batch_normalize ? createProgramInfoBatch(handler, groupConvProgramInfo.outputLayout, inputs, outputShape, activation) : null;
    return [groupConvProgramInfo].filter(x => !!x)
  }
  static createRunDatas(handler, texture, glProg, outTextureID) {
    const inputTDs = texture.map((t, i) => handler.getOrCreateTextureData(t, glProg[0].inputLayouts[i]));
    return [{
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(glProg[0].outputLayout, 'float32', outTextureID),
      uniformData: {}
    }];
  }
}
function createGroupConvProgramInfo(handler, inputs, outputShape, activation, batch_normalize) {
  const { funcActivation, nameActivation } = activation == "LINEAR" ? { funcActivation: ``, nameActivation: `` } : getGlActivation(activation)
  const hasBias = inputs.length > 2;
  const processBias = hasBias ? `value += getBias(output_channel);` : ``;
  const xShape = inputs[0].shape;
  const wShape = inputs[1].shape;
  const outputChannelsPerGroup = wShape[1];
  const inputChannelsPerGroup = wShape[0] / inputs[1].groups
  const glsl = getGlsl(handler.glContext.version);
  const samplers = hasBias ? ['X', 'W', 'Bias'] : ['X', 'W']
  const shaderSource = `
  const ivec2 strides = ivec2(${inputs[1].stride_x}, ${inputs[1].stride_y});
  const ivec2 pads = ivec2(${inputs[1].pad}, ${inputs[1].pad});
  ${funcActivation}
  void main() {
    ivec4 coords = getOutputCoords();
    int batch = coords.x;
    int output_channel = coords.y;
    ivec2 loc = coords.zw + pads;
    int group_id = output_channel / ${outputChannelsPerGroup};
    int wOutChannel = output_channel - group_id * ${outputChannelsPerGroup};
    float value = 0.0;
    for (int inChannelOffset = 0; inChannelOffset < ${inputChannelsPerGroup}; inChannelOffset++) {
      int input_channel = group_id * ${inputChannelsPerGroup} + inChannelOffset;
      for (int wWOff = 0; wWOff < ${wShape[2]}; wWOff++) {
        for (int wHOff = 0; wHOff < ${wShape[3]}; wHOff++) {
          ivec2 wOff = ivec2(wWOff * ${inputs[1].dilation}, wHOff * ${inputs[1].dilation});
          ivec2 wLoc = loc - wOff;
          ivec2 wLocIn = wLoc / strides;
          if (
            wLocIn * strides == wLoc &&
            wLocIn.x >= 0 && wLocIn.x < ${xShape[2]} &&
            wLocIn.y >= 0 && wLocIn.y < ${xShape[3]}
          ) {
            float xVal = getX(batch, input_channel, wLocIn.y, wLocIn.x);
            float wVal = getW(input_channel, wOutChannel, wHOff, wWOff);
            value += xVal * wVal;
          }
        }
      }
    }
    ${processBias}
    ${nameActivation}
    ${glsl.output} = vec4(value, .0, .0, .0);
  }
`;
  return {
    inputLayouts: inputs.map(t => handler.getOrCreateTextureLayout(t.TextureID, t.shape)),
    outputLayout: handler.createTextureLayoutFromShape(outputShape),
    samplers: samplers,
    shaderSource,
    hasMain: true,
  };
}