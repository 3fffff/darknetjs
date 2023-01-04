import { getGlsl } from "../libglsl/glsl-source.js"
export function sum(webgl, l) {
  const textures = [{ TextureID: "t" + (l.index - 1), shape: [l.batch, l.c, l.h, l.w] }, { TextureID: "t" + l.indexs, shape: [l.batch, l.c, l.h, l.w] }]
  const glProg = createProgramInfo(webgl, textures, [l.batch, l.out_c, l.out_h, l.out_w], l.type.toUpperCase())
  l.artifacts = [webgl.programManager.build(glProg)];
  l.runData = createRunData(webgl, textures, glProg, l.index)
}

function createProgramInfo(handler, inputs, outputShape, type) {
  const glsl = getGlsl(handler.glContext.version);
  const sumLine = inputs.map((v, i) => `${glsl.texture2D}(X${i},TexCoords)`).join(getSamOrShortcut(type));
  const samplers = inputs.map((v, i) => `X${i}`);
  return {
    inputLayouts: inputs.map(t => handler.getOrCreateTextureLayout(t.TextureID, t.shape)),
    outputLayout: handler.createTextureLayoutFromShape(outputShape),
    samplers,
    shaderSource: `
    void main() {
      vec4 result = ${sumLine};
      ${glsl.output} = result;
    }`,
    hasMain: true
  };
}

function getSamOrShortcut(type) {
  if (type == "SAM") return ' * '
  if (type == "SHORTCUT") return ' + '
  throw new Error("No recognized")
}