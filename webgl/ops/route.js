"use strict";
class WebGLRoute {
  static createProgramInfo(handler, inputs, outputShape, groups) {
    const rank = outputShape.length;
    // in most cases linear search is sufficient, as in most scenarios, only 2 tensors are concatenated
    const getTextureIndexWhereDataResidesMethod = WebGLRoute.getTextureIndexWhereDataResidesLinearSearch(inputs.length);
    const fetchDataFromCorrectTextureMethod = WebGLRoute.fetchDataFromCorrectTextureMethod(inputs.length, rank);
    const getValueFromArrayIndexMethod = WebGLRoute.getValueFromArrayIndexMethod(inputs.length);
    const samplers = inputs.map((v, i) => `X${i}`);
    const shaderSource = `
      ${fetchDataFromCorrectTextureMethod}
      ${getValueFromArrayIndexMethod}
      ${getTextureIndexWhereDataResidesMethod}
      float process(int indices[${rank}]) {
        int textureIndex = getTextureWhereDataResides (indices[${groups}]);

        if(textureIndex != 0) {
          indices[${groups}] = indices[${groups}] - int(getValueFromArrayIndex(sizeInConcatAxis, textureIndex-int(${groups})));
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
  static createRunData(handler) {
    const inputTDs = this.textures.map((t, i) => handler.getOrCreateTextureData(t, this.glProg.inputLayouts[i]));
    const sizeInConcatAxis = new Array(this.glProg.inputLayouts.length);
    let previousSum = 0;
    for (let i = 0; i < this.glProg.inputLayouts.length; ++i) {
      previousSum += this.glProg.inputLayouts[i].shape[1];
      sizeInConcatAxis[i] = previousSum;
    }
    const uniformData = { 'sizeInConcatAxis': sizeInConcatAxis };
    return [{
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(this.glProg.outputLayout, 'float32', "t" + this.index),
      uniformData
    }];
  }
  static getTextureIndexWhereDataResidesLinearSearch(numberOfTensors) {
    return `int getTextureWhereDataResides(int index) {
      for(int i=0; i<${numberOfTensors}; i++) {
          if(index < int(sizeInConcatAxis[i])){
              return i;
          }
        }
      }`;
  }
  static fetchDataFromCorrectTextureMethod(numberOfTensors, tensorRank) {
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
  static getValueFromArrayIndexMethod(arrayRank) {
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
}