"use strict";
/**
 * GLSL Library responsible for vec routines
 * Vec is an varible length int array. The length is fixed at the time of
 * generating the library functions from the dimensions of the output.
 */
class VecGlslLib extends GlslLib {
  constructor(context) {
    super(context);
  }
  getCustomTypes() {
    return {};
  }
  getFunctions() {
    return Object.assign(Object.assign(Object.assign(Object.assign({}, this.binaryVecFunctions()), this.copyVec()), this.setVecItem()), this.getVecItem());
  }
  binaryVecFunctions() {
    const outputLayout = this.context.programInfo.outputLayout;
    const rank = outputLayout.shape.length;
    const nameOp = { add: '+=', sub: '-=', mul: '*=', div: '/=' };
    const result = {};
    for (const name in nameOp) {
      const fname = `${name}Vec`;
      let assignmentBlock = '';
      for (let i = 0; i < rank; ++i) {
        assignmentBlock += `
          dest[${i}] ${nameOp[name]} src[${i}];
          `;
      }
      const body = `
        void ${fname}(int src[${rank}], out int dest[${rank}]) {
          ${assignmentBlock}
        }
        `;
      result[fname] = new GlslLibRoutine(body);
    }
    return result;
  }
  copyVec() {
    const outputLayout = this.context.programInfo.outputLayout;
    const rank = outputLayout.shape.length;
    let assignmentBlock = '';
    for (let i = 0; i < rank; ++i) {
      assignmentBlock += `
        dest[${i}] = src[${i}];
        `;
    }
    const body = `
      void copyVec(int src[${rank}], out int dest[${rank}]) {
        ${assignmentBlock}
      }
      `;
    return { copyVec: new GlslLibRoutine(body) };
  }
  setVecItem() {
    const outputLayout = this.context.programInfo.outputLayout;
    const rank = outputLayout.shape.length;
    let block = `
        if(index < 0)
            index =${rank} + index;
        if (index == 0)
            m[0] = value;
        `;
    for (let i = 1; i < rank - 1; ++i) {
      block += `
        else if (index == ${i})
            m[${i}] = value;
            `;
    }
    block += `
        else
            m[${rank - 1}] = value;
        `;
    const body = `
      void setVecItem(out int m[${rank}], int index, int value) {
        ${block}
      }
        `;
    return { setVecItem: new GlslLibRoutine(body) };
  }
  getVecItem() {
    const outputLayout = this.context.programInfo.outputLayout;
    const rank = outputLayout.shape.length;
    let block = `
        if(index < 0)
            index = ${rank} + index;
        if (index == 0)
            return m[0];
      `;
    for (let i = 1; i < rank - 1; ++i) {
      block += `
        else if (index == ${i})
            return m[${i}];
      `;
    }
    block += `
        else
            return m[${rank - 1}];
        `;
    const body = `
      int getVecItem(int m[${rank}], int index) {
        ${block}
      }
    `;
    return { getVecItem: new GlslLibRoutine(body) };
  }
}