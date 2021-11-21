"use strict";
class WebGLMatMul {
   static createProgramInfo(handler, l,inputs) {
        let sharedDim = inputs[0].shape[inputs[0].length - 1];
        let line = `value += _A(a) * _B(b);`;
        const rank = oShape.length;
        const shaderSource = `
      float process(int indices[${rank}]) {
          int a[${rank}];
          int b[${rank}];
          ${declareC}

          copyVec(indices, a);
          copyVec(indices, b);
          ${broadcastC}

          float value = 0.0;
          for (int k=0; k<${sharedDim}; ++k) {
              a[${rank - 1}] = k;
              b[${rank - 2}] = k;
              ${line}
          }
          return value;
      }`;
        const inputLayouts = inputs.map(t => handler.getOrCreateTextureLayout(t.TextureID,t.shape));
        console.log(inputLayouts)
        return {
            inputLayouts,
            outputLayout: handler.createTextureLayoutFromShape(oShape),
            samplers: ['A', 'B'],
            shaderSource,
        };
    }
    static createRunData(handler, inputs) {
        const inputTDs = inputs.map((t, i) => handler.getOrCreateTextureData(t, this.glProg.inputLayouts[i]));
        return {
            inputTextureDatas: inputTDs,
            outputTextureData: handler.createTextureDataFromLayout(this.glProg.outputLayout, "float32",this),
            uniformData: { }
        };
    }
}