export function route(webgl, l) {
  const textures = []
  let glProg;
  if (l.input_layers.length == 1) {
    textures.push({ groups: 0, TextureID: "t" + l.input_layers[0], shape: [l.batch, l.out_c, l.out_h, l.out_w] })
    glProg = createSplitProgramInfo(webgl, textures, [l.batch, l.out_c, l.out_h, l.out_w], l.group_id * l.out_c)
  } else {
    for (let i = 0; i < l.input_layers.length; ++i) textures.push({ groups: l.groups, TextureID: "t" + l.input_layers[i], shape: [l.batch, l.route_channels[i], l.out_h, l.out_w] })
    glProg = createProgramInfo(webgl, textures, [l.batch, l.out_c, l.out_h, l.out_w])
  }
  l.textures = textures
  l.artifacts = [webgl.programManager.build(glProg)];
  l.runData = createRunData(webgl, textures, glProg, l.index)
}

function createProgramInfo(handler, inputs, outputShape, axis = 1) {
  const rank = outputShape.length;
  // in most cases linear search is sufficient, as in most scenarios, only 2 tensors are concatenated
  const getTextureIndexWhereDataResides = getTextureIndexWhereDataResidesLinearSearch(inputs.length);
  const fetchDataFromCorrectTexture = fetchDataFromCorrectTextureMethod(inputs.length, rank);
  const getValueFromArrayIndex = getValueFromArrayIndexMethod(inputs.length);
  const samplers = inputs.map((v, i) => `X${i}`);
  const shaderSource = `
    ${fetchDataFromCorrectTexture}
    ${getValueFromArrayIndex}
    ${getTextureIndexWhereDataResides}
    float process(int indices[${rank}]) {
      int textureIndex = getTextureWhereDataResides (indices[${axis}]);

      if(textureIndex != 0) {
        indices[${axis}] = indices[${axis}] - int(getValueFromArrayIndex(sizeInConcatAxis, textureIndex-int(${axis})));
      }

      return fetchDataFromCorrectTexture(textureIndex, indices);
    }`;
  return {
    inputLayouts: inputs.map(t => handler.getOrCreateTextureLayout(t, t.shape)),
    outputLayout: handler.createTextureLayoutFromShape(outputShape),
    samplers,
    variables: [{ name: 'sizeInConcatAxis', type: 'int', arrayLength: inputs.length }],
    shaderSource,
  };
}
function createSplitProgramInfo(handler, input, outputShape, offset, axis = 1) {
  const rank = outputShape.length;
  const shaderSource = `
  float process(int indices[${rank}]) {
    indices[${axis}] += ${offset};
    return _A(indices);
  }`;
  return {
    inputLayouts: [handler.getOrCreateTextureLayout(input[0].TextureID, input[0].shape)],
    outputLayout: handler.createTextureLayoutFromShape(outputShape),
    samplers: ['A'],
    shaderSource,
  };
}
function createRunData(handler, textures, glProg, outTextureID) {
  const inputTDs = textures.map((t, i) => handler.getOrCreateTextureData(t, glProg.inputLayouts[i]));
  const sizeInConcatAxis = new Array(glProg.inputLayouts.length);
  let previousSum = 0;
  for (let i = 0; i < glProg.inputLayouts.length; ++i) {
    previousSum += glProg.inputLayouts[i].shape[1];
    sizeInConcatAxis[i] = previousSum;
  }
  const uniformData = { 'sizeInConcatAxis': sizeInConcatAxis };
  return [{
    inputTextureDatas: inputTDs,
    outputTextureData: handler.createTextureDataFromLayout(glProg.outputLayout, 'float32', "t" + outTextureID),
    uniformData
  }];
}
function getTextureIndexWhereDataResidesLinearSearch(numberOfTensors) {
  return `int getTextureWhereDataResides(int index) {
    for(int i=0; i<${numberOfTensors}; i++) {
        if(index < int(sizeInConcatAxis[i])){
            return i;
        }
      }
    }`;
}
function fetchDataFromCorrectTextureMethod(numberOfTensors, tensorRank) {
  const codeLines = [`float fetchDataFromCorrectTexture(int textureIndex, int indices[${tensorRank}]) {`];
  for (let i = 0; i < numberOfTensors; ++i) {
    if (i === 0) {
      codeLines.push(`\t` +
        `if (textureIndex == ${i}) { return _X${i}(indices); }`);
    }
    else if (i === numberOfTensors - 1) {
      codeLines.push(`\t` +
        `else { return _X${i}(indices); }`);
    }
    else {
      codeLines.push(`\t` +
        `else if (textureIndex == ${i}) { return _X${i}(indices); }`);
    }
  }
  codeLines.push(`\t` +
    `}`);
  return codeLines.join('\n');
}
function getValueFromArrayIndexMethod(arrayRank) {
  const codeLines = [`int getValueFromArrayIndex(int arr[${arrayRank}], int index) {`];
  for (let i = 0; i < arrayRank; ++i) {
    if (i === 0) {
      codeLines.push(`\t` +
        `if (index == ${i}) { return arr[${i}]; }`);
    }
    else if (i === arrayRank - 1) {
      codeLines.push(`\t` +
        `else { return arr[${i}]; }`);
    }
    else {
      codeLines.push(`\t` +
        `else if (index == ${i}) { return arr[${i}]; }`);
    }
  }
  codeLines.push(`\t` +
    `}`);
  return codeLines.join('\n');
}