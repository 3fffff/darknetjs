export function dropout(webgl, l) {
  l.runData = []
  l.runData.push({ outputTextureData: webgl.getTextureData("t" + (l.index - 1)) })
  webgl.setTextureData("t" + l.index, l.runData);
}