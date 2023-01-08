import { getGlActivation } from "./activation.js"

export function connected(webgl, l) {
  const textures = [{ TextureID: "tw" + l.index, shape: [l.weights.length, 1], output: l.weights }, { TextureID: "t" + (l.index - 1), shape: [1, l.weights.length] }]
  const glProg = createProgramInfo(webgl, l)
  l.artifacts = [webgl.programManager.build(glProg)];
  l.runData = createRunData(webgl, textures, glProg, l.index)
}

function createProgramInfo(handler, inputs, outputShape) {
  const { funcActivation, nameActivation } = activation == "LINEAR" ? { funcActivation: ``, nameActivation: `` } : getGlActivation(activation)
  const hasBias = inputs.length > 2;
  const processBias = hasBias ? `value += getBias();` : ``;
  const sharedDim = inputs[0].shape[inputs[0].length - 1];
  const rank = outputShape.length;
  const shaderSource = `
  ${funcActivation}
    float process(int indices[${rank}]) {
        int a[${rank}];
        int b[${rank}];

        copyVec(indices, a);
        copyVec(indices, b);

        float value = 0.0;
        for (int k=0; k<${sharedDim}; ++k) {
            a[${rank - 1}] = k;
            b[${rank - 2}] = k;
            value += _A(a) * _B(b);
        }
        ${processBias}
        ${nameActivation}
        return value;
    }`;
  const inputLayouts = inputs.map(t => handler.getOrCreateTextureLayout(t.TextureID, t.shape));
  return {
    inputLayouts,
    outputLayout: handler.createTextureLayoutFromShape(outputShape),
    samplers: ['A', 'B'],
    shaderSource,
  };
}
function createRunData(handler, textures, glProg, outTextureID) {
  const inputTDs = textures.map((t, i) => handler.getOrCreateTextureData(t, glProg.inputLayouts[i]));
  return {
    inputTextureDatas: inputTDs,
    outputTextureData: handler.createTextureDataFromLayout(glProg.outputLayout, "float32", "t" + outTextureID),
    uniformData: {}
  };
}