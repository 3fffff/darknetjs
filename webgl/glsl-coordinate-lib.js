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
    return Object.assign(Object.assign(Object.assign(Object.assign(Object.assign({}, this.offsetToCoords()), this.coordsToOffset()), this.toVec()), this.valueFrom()), this.GetCommonUtilFuncs());
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
    let result = {};
    let funcName = "uvFromFlat";
    result[funcName] = new GlslLibRoutine("\n    vec2 uvFromFlat(int texNumR, int texNumC, int index) {\n      int texC = index / texNumR;\n      int texR = index - texC * texNumR;\n      // TODO: swap texR, texC order in following function so row is corresponding to u and column is corresponding to v.\n      return (vec2(texR, texC) + halfCR) / vec2(texNumR, texNumC);\n    }\n    ");
    funcName = "packedUVfrom1D";
    result[funcName] = new GlslLibRoutine("\n      vec2 packedUVfrom1D(int texNumR, int texNumC, int index) {\n        int texelIndex = index / 2;\n        int texR = texelIndex / texNumC;\n        int texC = texelIndex - texR * texNumC;\n        return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);\n      }\n      ");
    funcName = "packedUVfrom2D";
    result[funcName] = new GlslLibRoutine("\n      vec2 packedUVfrom2D(int texNumR, int texNumC, int texelsInLogicalRow, int row, int col) {\n        int texelIndex = (row / 2) * texelsInLogicalRow + (col / 2);\n        int texR = texelIndex / texNumC;\n        int texC = texelIndex - texR * texNumC;\n        return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);\n      }\n      ");
    funcName = "packedUVfrom3D";
    result[funcName] = new GlslLibRoutine("\n      vec2 packedUVfrom3D(int texNumR, int texNumC,\n          int texelsInBatch, int texelsInLogicalRow, int b,\n          int row, int col) {\n        int index = b * texelsInBatch + (row / 2) * texelsInLogicalRow + (col / 2);\n        int texR = index / texNumC;\n        int texC = index - texR * texNumC;\n        return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);\n      }\n      ");
    funcName = "sampleTexture";
    let glsl = getGlsl(this.context.glContext.version);
    result[funcName] = new GlslLibRoutine("\n        float sampleTexture(sampler2D textureSampler, vec2 uv) {\n            return " + glsl.texture2D + "(textureSampler, uv).r;\n        }");
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
    return { getOutputCoords: source };
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