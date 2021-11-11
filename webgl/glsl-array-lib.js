"use strict";
/**
 * This library produces routines needed for non-constant access to uniform arrays
 */
class ArrayGlslLib extends GlslLib {
  getFunctions() {
    return this.generate();
  }
  getCustomTypes() {
    return {};
  }
  constructor(context) {
    super(context);
  }
  generate() {
    const result = {};
    for (let i = 1; i <= 16; i++) {
      result[`setItem${i}`] = new GlslLibRoutine(this.generateSetItem(i));
      result[`getItem${i}`] = new GlslLibRoutine(this.generateGetItem(i));
    }
    return result;
  }
  generateSetItem(length) {
    let block = `
       if(index < 0)
           index = ${length} + index;
       if (index == 0)
           a[0] = value;
       `;
    for (let i = 1; i < length - 1; ++i) {
      block += `
       else if (index == ${i})
           a[${i}] = value;
           `;
    }
    block += `
       else
           a[${length - 1}] = value;
       `;
    const body = `
     void setItem${length}(out float a[${length}], int index, float value) {
       ${block}
     }
       `;
    return body;
  }
  generateGetItem(length) {
    let block = `
       if(index < 0)
           index = ${length} + index;
       if (index == 0)
           return a[0];
     `;
    for (let i = 1; i < length - 1; ++i) {
      block += `
       else if (index == ${i})
           return a[${i}];
     `;
    }
    block += `
       else
           return a[${length - 1}];
       `;
    const body = `
     float getItem${length}(float a[${length}], int index) {
       ${block}
     }
   `;
    return body;
  }
}