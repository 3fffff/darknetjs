"use strict";
/**
 * GLSL Library responsible for data types and routines for manipulating
 * coordinates and mapping to/from tensor indices
 */
class ShapeUtilsGlslLib extends GlslLib {
  constructor(context) {
    super(context);
  }
  getFunctions() {
    return Object.assign({}, this.bcastIndex(), this.bcastMatmulIndex(), this.offsetToIndices(), this.indicesToOffset(), this.incrementIndices());
  }
  bcastIndex() {
    const programInfo = this.context.programInfo;
    const outputRank = programInfo.outputLayout.shape.length;
    const result = {};
    this.context.programInfo.samplers.forEach((name, i) => {
      const shape = programInfo.inputLayouts[i].shape;
      if (shape.length <= outputRank) {
        const rank = shape.length;
        const dimOffset = outputRank - rank;
        const funcName = `bcastIndices_${name}`;
        let block = '';
        for (let i = 0; i < rank; ++i) {
          block += `
          realIndices[${i}] = int( mod(float(bcastedIndices[${dimOffset + i}]), ${shape[i]}.0) );
          `;
        }
        const body = `
        void ${funcName} (int bcastedIndices[${outputRank}], out int realIndices[${rank}]) {
          ${block}
        }
        `;
        result[funcName] = new GlslLibRoutine(body);
      }
    });
    return result;
  }
  bcastMatmulIndex() {
    const programInfo = this.context.programInfo;
    const outputRank = programInfo.outputLayout.shape.length;
    const result = {};
    this.context.programInfo.samplers.forEach((name, i) => {
      const shape = programInfo.inputLayouts[i].shape;
      if (!(shape.length < 2 || shape.length > outputRank)) {
        const rank = shape.length;
        const dimOffset = outputRank - rank;
        const funcName = `bcastMatmulIndices_${name}`;
        let block = '';
        for (let i = 0; i < rank - 2; ++i) {
          block += `
          realIndices[${i}] = int( mod(float(bcastedIndices[${dimOffset + i}]), ${shape[i]}.0) );
          `;
        }
        const body = `
        void ${funcName}(int bcastedIndices[${outputRank}], out int realIndices[${rank}]) {
          ${block}
          realIndices[${rank - 1}] = bcastedIndices[${outputRank - 1}];
          realIndices[${rank - 2}] = bcastedIndices[${outputRank - 2}];
        }
        `;
        result[funcName] = new GlslLibRoutine(body);
      }
    });
    return result;
  }
  indicesToOffset() {
    const programInfo = this.context.programInfo;
    const result = {};
    this.context.programInfo.samplers.forEach((name, i) => {
      const shape = programInfo.inputLayouts[i].shape;
      const strides = programInfo.inputLayouts[i].strides;
      const rank = shape.length;
      let funcName = `indicesToOffset_${name}`;
      result[funcName] = new GlslLibRoutine(ShapeUtilsGlslLib.indexToOffsetSingle(funcName, rank, strides));
      funcName = `indicesToOffset_${name}_T`;
      result[funcName] =
        new GlslLibRoutine(ShapeUtilsGlslLib.indexToOffsetSingle(funcName, rank, strides.slice().reverse()));
    });
    return result;
  }
  static indexToOffsetSingle(name, rank, strides) {
    let block = '';
    for (let i = rank - 1; i >= 0; --i) {
      block += `
        offset += indices[${i}] * ${strides[i]};
        `;
    }
    return `
      int ${name}(int indices[${rank}]) {
        int offset = 0;
        ${block}
        return offset;
      }
      `;
  }
  offsetToIndices() {
    const programInfo = this.context.programInfo;
    const result = {};
    this.context.programInfo.samplers.forEach((name, i) => {
      const shape = programInfo.inputLayouts[i].shape;
      const strides = programInfo.inputLayouts[i].strides;
      const rank = shape.length;
      let funcName = `offsetToIndices_${name}`;
      result[funcName] = new GlslLibRoutine(ShapeUtilsGlslLib.offsetToIndicesSingle(funcName, rank, strides));
      funcName = `offsetToIndices_${name}_T`;
      result[funcName] =
        new GlslLibRoutine(ShapeUtilsGlslLib.offsetToIndicesSingle(funcName, rank, strides.slice().reverse()));
    });
    return result;
  }
  static offsetToIndicesSingle(name, rank, strides) {
    const stridesBlock = [];
    for (let i = 0; i < rank - 1; ++i) {
      stridesBlock.push(`
      indices[${i}] = offset / ${strides[i]};`);
      stridesBlock.push(`
        offset -= indices[${i}] * ${strides[i]};`);
    }
    stridesBlock.push(`
      indices[${rank - 1}] = offset;`);
    return `
      void ${name}(int offset, out int indices[${rank}]) {
        ${stridesBlock.join('')}
      }
      `;
  }
  incrementIndices() {
    const programInfo = this.context.programInfo;
    const result = {};
    this.context.programInfo.samplers.forEach((name, i) => {
      const shape = programInfo.inputLayouts[i].shape;
      const rank = shape.length;
      const funcName = `incrementIndices_${name}`;
      let shapeInit = '';
      for (let i = 0; i < rank; ++i) {
        shapeInit += `
        shape[${i}] = ${shape[i]};`;
      }
      const body = `
        void ${funcName}(int axis, out int indices[${rank}]) {
          int shape[${rank}];
          ${shapeInit};
          for(int i = ${rank} -1 ; i >= 0; --i) {
            if(i > axis) continue;
            indices[i] += 1;
            if(indices[i] < shape[i]) {
              break;
            }
            indices[i] = 0;
          }
        }
        `;
      result[funcName] = new GlslLibRoutine(body);
    });
    return result;
  }
}