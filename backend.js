class Backend {
  constructor(layers, wasm, webgl) {
    this.layers = layers
  }
  initWasm() {
    const wasmSupport = wasmcheck.support()
    const Simd = wasmcheck.feature.simd
    const thread = wasmcheck.feature.threads
    if (!wasmSupport) throw new Error("wasm is not supported")
    if (!Simd) throw new Error("simd is not supported")
    if (wasmSupport && thread) {
      initWorkers(3).then(
        () => {
          WasmBinding.getInstance();
          postMessage({ type: 'init-success' });
        },
      );
    }
    else if (wasmSupport && !thread) {
      initWorkers(0).then(
        () => {
          WasmBinding.getInstance();
          postMessage({ type: 'init-success' });
        },
      );
    }
    for (let i = 1; i < this.layers.length; i++) {
      if (this.layers[i].type.toUpperCase() == 'CONVOLUTIONAL') this.layers[i].forward = WasmConv
      else if (this.layers[i].type.toUpperCase() == 'MAXPOOL') this.layers[i].forward = WasmPool
    }
  }
  initGL() {
    ////////////////////////////////////////////simple testing
    this.webgl = new WebGL("webgl2")
    for (let i = 1; i < this.layers.length; i++) {
      const l = this.layers[i]
      l.TextureID = "t" + l.index
      if (l.type.toUpperCase() == 'CONVOLUTIONAL') {
        l.glInputs = [{ TextureID: "t" + (l.index - 1), activation: l.activation, pad: l.pad, size: l.size, shape: [l.batch, l.c, l.h, l.w], output: this.layers[i - 1].output },
        { batch: l.batch, filters: l.filters, size: l.size, weights: l.weights, dilation: l.dilation, TextureID: 'w' + l.index, groups: l.groups, shape: [l.c, l.filters, l.size, l.size], pad: l.pad, stride_x: l.stride_x, stride_y: l.stride_y },
        { output: l.bias, TextureID: 'bias' + l.index, shape: [l.filters] }]
        l.glProg = WebGLConv.createProgramInfos(this.webgl, l.glInputs, [l.batch, l.out_c, l.out_h, l.out_w])
        l.glData = WebGLConv.createRunDatas
        l.artifacts = []
        l.artifacts.push(this.webgl.programManager.build(l.glProg[0]));
        l.artifacts.push(this.webgl.programManager.build(l.glProg[1]));
        if (l.glProg[2] != null) l.artifacts.push(this.webgl.programManager.build(l.glProg[2]));
        this.webgl.programManager.setArtifact("t" + l.index, l.artifacts[0]);
        this.webgl.programManager.setArtifact("t" + l.index, l.artifacts[1]);
        if (l.glProg[2] != null) this.webgl.programManager.setArtifact("t" + l.index, l.artifacts[2]);
        l.runData = l.glData(this.webgl, l.glInputs);
      }
      else if (l.type.toUpperCase() == 'MAXPOOL') {
        const glInputs = { TextureID: "t" + (l.index - 1), pad: l.pad, size: l.size, stride_x: l.stride_x, stride_y: l.stride_x, shape: [l.batch, l.c, l.h, l.w], output: this.layers[i - 1].output }
        const glProg = WebGLPool.createProgramInfo(this.webgl, glInputs, [l.batch, l.out_c, l.out_h, l.out_w])
        l.artifacts = [this.webgl.programManager.build(glProg)]
        this.webgl.programManager.setArtifact(l.TextureID, l.artifacts[0]);
        l.runData = WebGLPool.createRunData(this.webgl, glInputs, glProg, l);
      }
      else if (l.type.toUpperCase() == 'UPSAMPLE') {
        const glInputs = { TextureID: "t" + (l.index - 1), scale: 1, stride: l.stride, shape: [l.batch, l.c, l.h, l.w], output: this.layers[i - 1].output }
        const glProg = WebGLUpsample.createProgramInfo(this.webgl, glInputs, [l.batch, l.out_c, l.out_h, l.out_w])
        l.artifacts = [this.webgl.programManager.build(glProg)];
        this.webgl.programManager.setArtifact(l.TextureID, l.artifacts[0]);
        l.runData = WebGLUpsample.createRunData(this.webgl, glInputs, glProg, l);
      }
      else if (l.type.toUpperCase() == 'ROUTE') {
        const glInputs = []
        if (l.input_layers.length == 1) glInputs.push({ groups: l.groups, TextureID: "t" + (l.index - 1), shape: [l.input_layers[0].batch, l.out_c, l.input_layers[0].h, l.input_layers[0].w], output: this.layers[l.input_layers[0].index].output })
        else for (let i = 0; i < l.input_layers.length; ++i) glInputs.push({ groups: l.groups, TextureID: "t" + l.index, shape: [l.input_layers[i].batch, l.input_layers[i].c, l.input_layers[i].h, l.input_layers[i].w], output: this.layers[l.input_layers[i].index].output })
        const glProg = WebGLRoute.createProgramInfo(this.webgl, glInputs, [l.batch, l.out_c, l.out_h, l.out_w], 1)
        l.artifacts = [this.webgl.programManager.build(glProg)];
        this.webgl.programManager.setArtifact(l.TextureID, l.artifacts[0]);
        l.runData = WebGLRoute.createRunData(this.webgl, glInputs, glProg, l);
      }
      else if (l.type.toUpperCase() == 'SHORTCUT') {
        l.output = layers[this.index - 1].output;
      } else if (l.type.toUpperCase() == 'SAM') {
        l.output = layers[this.index - 1].output;
      }
    }
  }
  async start(img) {
    this.layers[0].output = ImageProcess.resize_image(img, this.layers[0].w, this.layers[0].h, 768, 576, 3)
    for (let i = 1; i < this.layers.length; ++i)await this.layers[i].forward(this.layers)
  }
  run(img) {
    this.layers[0].output = ImageProcess.resize_image(img, this.layers[0].w, this.layers[0].h, 768, 576, 3)
    this.layers[1].glInputs[0].output = this.layers[0].output
    for (let i = 1; i < 2; ++i) {
      for (let j = 0; j < this.layers[i].artifacts.length; j++) {
        this.layers[i].runData = this.layers[i].glData(this.webgl, this.layers[i].glInputs);
        this.webgl.programManager.run(this.layers[i].artifacts[j], this.layers[i].runData[j]);
        console.log(this.layers[i].runData[j].outputTextureData.gldata())
      }
    }
  }
}
