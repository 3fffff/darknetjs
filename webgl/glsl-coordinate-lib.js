"use strict";
/**
 * GLSL Library responsible for data types and routines for manipulating
 * coordinates and mapping to/from tensor indices
 */
class CoordsGlslLib extends GlslLib {
  constructor(context) {
    super(context);
  }
  getFunctions() {
    return Object.assign({}, this.offsetToCoords(), this.coordsToOffset(), this.toVec(), this.valueFrom(), this.GetCommonUtilFuncs(), this.getInputsSamplingSnippets(), this.getOutputSamplingSnippet());
  }
  /**
   * Produces a function that can map from
   * 2D normalzied coordinates (s,t) to a flat offset
   */
  offsetToCoords() {
    const funcName = `offsetToCoords`;
    return {
      offsetToCoords: new GlslLibRoutine(`
      vec2 ${funcName}(int offset, int width, int height) {
        int t = offset / width;
        int s = offset - t*width;
        vec2 coords = (vec2(s,t) + vec2(0.5,0.5)) / vec2(width, height);
        return coords;
      }
      `)
    };
  }
  /**
 * Generates code for common UV coords computation utility functions.
 */
  GetCommonUtilFuncs() {
    const result = {};
    result["uvFromFlat"] = new GlslLibRoutine(`vec2 uvFromFlat(int texNumR, int texNumC, int index) {int texC = index / texNumR;  int texR = index - texC * texNumR; return (vec2(texR, texC) + halfCR) / vec2(texNumR, texNumC);   }`);
    result["packedUVfrom1D"] = new GlslLibRoutine(`vec2 packedUVfrom1D(int texNumR, int texNumC, int index) { int texelIndex = index / 2; int texR = texelIndex / texNumC; int texC = texelIndex - texR * texNumC;       return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);  }  `);
    result["packedUVfrom2D"] = new GlslLibRoutine(`vec2 packedUVfrom2D(int texNumR, int texNumC, int texelsInLogicalRow, int row, int col) { int texelIndex = (row / 2) * texelsInLogicalRow + (col / 2);       int texR = texelIndex / texNumC;  int texC = texelIndex - texR * texNumC;  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR); }`);
    result["packedUVfrom3D"] = new GlslLibRoutine(`vec2 packedUVfrom3D(int texNumR, int texNumC, int texelsInBatch, int texelsInLogicalRow, int b, int row, int col) { int index = b * texelsInBatch + (row / 2) * texelsInLogicalRow + (col / 2);  int texR = index / texNumC; int texC = index - texR * texNumC;  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);}`);
    const glsl = getGlsl(this.context.glContext.version);
    result["sampleTexture"] = new GlslLibRoutine(`float sampleTexture(sampler2D textureSampler, vec2 uv) { return ${glsl.texture2D}(textureSampler, uv).r; }`);
    return result;
  };
  /**
 * Generates code for output sampler.
 */
  getOutputUnpacked4DCoords(shape, texShape) {
    const funcName = `getOutputCoords`;
    let source = '';
    const rank = shape.length;

    let strides = null;
    if (rank < 2) strides = [];
    strides = new Array(rank - 1);
    strides[rank - 2] = shape[rank - 1];
    for (let i = rank - 3; i >= 0; --i) strides[i] = strides[i + 1] * shape[i + 1];
    const coordsToCompute = ['r', 'c', 'd', 'd2'];
    const coordsFromIndexSnippet = strides
      .map((stride, i) => {
        const line1 = `int ${coordsToCompute[i]} = index / ${stride}`;
        const line2 = i === strides.length - 1 ?
          `int ${coordsToCompute[i + 1]} = index - ${coordsToCompute[i]} * ${stride}` :
          `index -= ${coordsToCompute[i]} * ${stride}`;
        return `${line1}; ${line2};`;
      })
      .join('');

    source = `ivec4 ${funcName}() {
      ivec2 resTexRC = ivec2(TexCoords.xy *
                            vec2(${texShape[0]}, ${texShape[1]}));
      int index = resTexRC.y * ${texShape[0]} + resTexRC.x;
      ${coordsFromIndexSnippet}
      return ivec4(r, c, d, d2);
    }`;
    return new GlslLibRoutine(source);
  }

  getUnpackedSampler4D(funcName, name, inputLayout) {
    const shape = inputLayout.shape;
    const stride2 = shape[3];
    const stride1 = shape[2] * stride2;
    const stride0 = shape[1] * stride1;
    const texNumR = inputLayout.width;
    const texNumC = inputLayout.height;
    const source = `float ${funcName} (int row, int col, int depth, int depth2) {
              int index = row * ${stride0} + col * ${stride1} + depth2 * ${stride2} + depth;
              vec2 uv = uvFromFlat(${texNumR}, ${texNumC}, index);
              return sampleTexture(${name}, uv); }`;
    return new GlslLibRoutine(source, ['coordinates.uvFromFlat', 'coordinates.sampleTexture', 'coordinates.coordsToOffset']);
  };

  getUnpackedOutputSamplingSnippet(outputLayout) {
    const outShape = outputLayout.unpackedShape;
    const outTexShape = [outputLayout.width, outputLayout.height];
    const result = {};
    const funcName = 'getOutputCoords';
    result[funcName] = this.getOutputUnpacked4DCoords(outShape, outTexShape);
    const glsl = getGlsl(this.context.glContext.version);
    const floatTextureSetRSource = `void setOutput(float val) {${glsl.output} = vec4(val, 0, 0, 0);}`;
    const floatTextureSetRFuncName = 'floatTextureSetR';
    result[floatTextureSetRFuncName] = new GlslLibRoutine(floatTextureSetRSource);
    return result;
  };
  getInputsSamplingSnippets() {
    const self = this;
    const result = {};
    const outputLayout = this.context.programInfo.outputLayout;
    this.context.programInfo.samplers.forEach(function (samplerName, i) {
      const inputLayout = self.context.programInfo.inputLayouts[i];
      const funcName = self.generateShaderFuncNameFromInputSamplerName(samplerName);
      result[funcName] = self.getUnpackedSamplerFromInput(funcName, samplerName, inputLayout);
      const outCoordFuncName = self.generateShaderFuncNameFromInputSamplerNameAtOutCoords(samplerName);
      if (inputLayout.unpackedShape.length <= outputLayout.unpackedShape.length) {
        result[outCoordFuncName] = self.getUnpackedSamplerAtOutputCoords(outCoordFuncName, inputLayout, outputLayout, samplerName);
      }
    });
    return result;
  };
  getUnpackedSamplerFromInput(funcName, name, inputLayout) {
    const shape = inputLayout.shape;
    switch (shape.length) {
      case 1:return this.getUnpackedSampler1D(funcName, name, inputLayout);
      case 2: null;
      case 3: null;
      case 4: return this.getUnpackedSampler4D(funcName, name, inputLayout);
      default: throw new Error("Unsupported dimension " + shape.length + "-D");
    }
  };
  getUnpackedSampler1D(funcName, name, inputLayout) {
    const tNumR = inputLayout.width;
    const tNumC = inputLayout.height;
    if (tNumC === 1 && tNumR === 1) {
      const source_4 = `float ${funcName}(int index) { return sampleTexture(${name}, halfCR);}`;
      return new GlslLibRoutine(source_4, ['coordinates.sampleTexture']);
    }
    if (tNumC === 1) {
      const source_5 = `float ${funcName}(int index) { vec2 uv = vec2((float(index) + 0.5) / ${tNumR}.0, 0.5);    return sampleTexture(${name}, uv);  }`;
      return new GlslLibRoutine(source_5, ['coordinates.sampleTexture']);
    }
    if (tNumR === 1) {
      const source_6 = `float ${funcName}(int index) { vec2 uv = vec2(0.5, (float(index) + 0.5) / ${tNumC}.0);    return sampleTexture(${name}, uv);  }`;
      return new GlslLibRoutine(source_6, ['coordinates.sampleTexture']);
    }
    const source = `float ${funcName}(int index) { vec2 uv = uvFromFlat(${tNumR}, ${tNumC}, index);  return sampleTexture(${name}, uv);}`;
    return new GlslLibRoutine(source, ['coordinates.uvFromFlat', 'coordinates.sampleTexture']);
  };

  getUnpackedSamplerAtOutputCoords(funcName, inputLayout, outputLayout, name) {
    const outTexShape = [outputLayout.width, outputLayout.height];
    const inTexShape = [inputLayout.width, inputLayout.height];
    const inRank = inputLayout.unpackedShape.length;
    const outRank = outputLayout.unpackedShape.length;
    const inShape = inputLayout.unpackedShape;
    const outShape = outputLayout.unpackedShape;
    const texFuncSnippet = this.generateShaderFuncNameFromInputSamplerName(name);
    const arraysEqual = function (n1, n2) {
      if (n1.length !== n2.length) return false;
      for (let i = 0; i < n1.length; i++)
        if (n1[i] !== n2[i])
          return false;
      return true;
    };
    if (inRank === outRank && arraysEqual(inTexShape, outTexShape)) {
      const source_1 = `float ${funcName} () {    return sampleTexture(${name},TexCoords);}`;
      return new GlslLibRoutine(source_1, ['coordinates.sampleTexture']);
    }
    const getBroadcastDims = function (inputShape, outputShape) {
      const inRank = inputShape.length;
      const dims = [];
      for (let i = 0; i < inRank; i++) {
        const dim = inRank - 1 - i;
        const a = inputShape[dim] || 1;
        const b = outputShape[outputShape.length - 1 - i] || 1;
        if (b > 1 && a === 1) dims.unshift(dim);
      }
      return dims;
    };
    const type = 'ivec4';
    const broadcastDims = getBroadcastDims(inShape, outShape);
    const rankDiff = outRank - inRank;
    let coordsSnippet;
    const fields = ['x', 'y', 'z', 'w', 'u', 'v'];
    if (inRank === 0) {
      coordsSnippet = '';
    }
    else if (outRank < 2 && broadcastDims.length >= 1) {
      coordsSnippet = 'coords = 0;';
    }
    else {
      coordsSnippet = broadcastDims.map(function (d) { return "coords." + fields[d + rankDiff] + " = 0;"; }).join('');
    }
    let unpackedCoordsSnippet = '';
    if (outRank < 2 && inRank > 0) {
      unpackedCoordsSnippet = 'coords';
    }
    else {
      unpackedCoordsSnippet = inputLayout.unpackedShape.map(function (s, i) { return "coords." + fields[i + rankDiff]; }).join(', ');
    }
    const source = `float ${funcName} () { ${type} coords = getOutputCoords();  ${coordsSnippet} return ${texFuncSnippet} ( ${unpackedCoordsSnippet} );}`;
    return new GlslLibRoutine(source, ['coordinates.getOutputCoords']);
  };
  getOutputSamplingSnippet() {
    const outputLayout = this.context.programInfo.outputLayout;
    return this.getUnpackedOutputSamplingSnippet(outputLayout);
  };
  generateShaderFuncNameFromInputSamplerName(samplerName) {
    if(typeof samplerName !== 'undefined' && samplerName.length === 0)
      throw new Error('empty string found for sampler name'); 
    return 'get' + samplerName.charAt(0).toUpperCase() + samplerName.slice(1);
  }
  generateShaderFuncNameFromInputSamplerNameAtOutCoords(samplerName) {
    if(typeof samplerName !== 'undefined' && samplerName.length === 0)
      throw new Error('empty string found for sampler name'); 
    return 'get' + samplerName.charAt(0).toUpperCase() + samplerName.slice(1) + 'AtOutCoords';
  }
  /**
   * Produces a function that can map from
   * 2D normalzied coordinates (s,t) to a flat offset
   */
  coordsToOffset() {
    const funcName = `coordsToOffset`;
    return {
      coordsToOffset: new GlslLibRoutine(`
      int ${funcName}(vec2 coords, int width, int height) {
        float s = coords.s * float(width);
        float t = coords.t * float(height);
        int offset = int(t) * width + int(s);
        return offset;
      }
      `)
    };
  }
  /**
   * This is the main function to map from the given texture coordiantes (s,t)
   * to logical indices for the output
   * There will only be one single variation of this
   * Also see coordsToOffset and offsetToIndices for input-specific versions
   */
  toVec() {
    const output = this.context.programInfo.outputLayout;
    const rank = output.shape.length;
    const strides = output.strides;
    const xScale = output.width;
    const yScale = output.height;
    const stridesBlock = [];
    for (let i = 0; i < rank - 1; ++i) {
      stridesBlock.push(`
        c[${i}] = offset / ${strides[i]};`);
      stridesBlock.push(`
        offset -= c[${i}] * ${strides[i]};`);
    }
    stridesBlock.push(`
        c[${rank - 1}] = offset;`);
    const body = `
      void toVec(vec2 texCoords, out int c[${rank}]) {
        int offset = coordsToOffset(texCoords, ${xScale}, ${yScale});
        ${stridesBlock.join('')}
      }
      void toVec(int offset, out int c[${rank}]) {
        ${stridesBlock.join('')}
      }
    `;
    return { toVec: new GlslLibRoutine(body, ['coordinates.coordsToOffset']) };
  }
  /**
   * These are value getter functions generated for each input
   * Each function is hardwired to the name and dimensions of the input
   * An '_T' variation is also produced which accesses values as if the
   * input was transposed
   */
  valueFrom() {
    const programInfo = this.context.programInfo;
    const result = {};
    this.context.programInfo.samplers.forEach((name, i) => {
      const layout = programInfo.inputLayouts[i];
      const shape = layout.shape;
      const rank = shape.length;
      let funcName = `_${name}`;
      result[funcName] = new GlslLibRoutine(this.getValueFromSingle(name, rank, layout.width, layout.height, false), [`shapeUtils.indicesToOffset${funcName}`, `coordinates.offsetToCoords`, `fragcolor.getColorAsFloat`]);
      funcName = funcName + '_T';
      result[funcName] = new GlslLibRoutine(this.getValueFromSingle(name, rank, layout.width, layout.height, true), [`shapeUtils.indicesToOffset${funcName}`, `coordinates.offsetToCoords`, `fragcolor.getColorAsFloat`]);
    });
    return result;
  }
  /**
   * Produces one value getter function for the name and rank given
   * If a transpose is set proper offsetToCoords mapping will be used
   * @param name name of the function
   * @param rank rank of the input
   * @param transpose whether or not should generate a transpose variation
   */
  getValueFromSingle(varName, rank, width, height, transpose) {
    let name = `_${varName}`;
    if (transpose) {
      name = name + '_T';
    }
    const glsl = getGlsl(this.context.glContext.version);
    return `
        float ${name}(int m[${rank}]) {
          int offset = indicesToOffset${name}(m);
          vec2 coords = offsetToCoords(offset, ${width}, ${height});
          float value = getColorAsFloat(${glsl.texture2D}(${varName}, coords));
          return value;
        }
        `;
  }
}