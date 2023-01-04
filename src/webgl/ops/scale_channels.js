export function scale_channels(webgl, l) {
  const textures = []
  textures.push({ groups: l.groups, TextureID: "t" + l.input_layers[i].index, shape: [l.input_layers[i].batch, l.input_layers[i].out_c, l.input_layers[i].out_h, l.input_layers[i].out_w] })
  const glProg = createScaleChannelsProgramInfo(webgl, textures, [l.batch, l.out_c, l.out_h, l.out_w])
  l.artifacts = [webgl.programManager.build(glProg)];
  l.runData = createRunData(webgl, textures, glProg, l.index)
}

function createScaleChannelsProgramInfo(handler, inputs, outputShape) {
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
function createRunData(handler, textures, glProg, outTextureID) {
  const inputTDs = textures.map((t, i) => handler.getOrCreateTextureData(t, glProg.inputLayouts[i]));
  return [{
    inputTextureDatas: inputTDs,
    outputTextureData: handler.createTextureDataFromLayout(glProg.outputLayout, 'float32', "t" + outTextureID),
    uniformData: {}
  }];
}