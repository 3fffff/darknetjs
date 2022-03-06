class Backend {
  constructor(layers) {
    this.layers = layers;
    this.first = false
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
    this.webgl = new WebGL("webgl2")
    for (let i = 1; i < this.layers.length; i++) {
      const l = this.layers[i]
      if (l.type.toUpperCase() == 'CONVOLUTIONAL') {
        l.textures = [{ index: l.index, activation: l.activation, TextureID: "t" + (l.index - 1), activation: l.activation, pad: l.pad, size: l.size, shape: [l.batch, l.c, l.h, l.w] },
        { batch: l.batch, filters: l.filters, size: l.size, weights: l.weights, dilation: l.dilation, TextureID: 'w' + l.index, groups: l.groups, shape: [l.filters, l.c, l.size, l.size], pad: l.pad, stride_x: l.stride_x, stride_y: l.stride_y },
        { output: l.biases, TextureID: 'bias' + l.index, shape: [l.filters] }]
        l.glProg = WebGLConv.createProgramInfos(this.webgl, l.textures, [l.batch, l.out_c, l.out_h, l.out_w])
        l.artifacts = []
        l.artifacts.push(this.webgl.programManager.build(l.glProg[0]));
        l.artifacts.push(this.webgl.programManager.build(l.glProg[1]));
        if (l.glProg[2] != null) l.artifacts.push(this.webgl.programManager.build(l.glProg[2]));
        l.createRunData = WebGLConv.createRunDatas
      }
      else if (l.type.toUpperCase() == 'MAXPOOL') {
        l.textures = [{ TextureID: "t" + (l.index - 1), pad: l.pad, size: l.size, stride_x: l.stride_x, stride_y: l.stride_x, shape: [l.batch, l.c, l.h, l.w] }]
        l.glProg = WebGLPool.createProgramInfo(this.webgl, l.textures[0], [l.batch, l.out_c, l.out_h, l.out_w])
        l.artifacts = [this.webgl.programManager.build(l.glProg)]
        l.createRunData = WebGLPool.createRunData
      }
      else if (l.type.toUpperCase() == 'UPSAMPLE') {
        l.textures = [{ TextureID: "t" + (l.index - 1), scale: 1, stride: l.stride, shape: [l.batch, l.c, l.h, l.w] }]
        l.glProg = WebGLUpsample.createProgramInfo(this.webgl, l.textures[0], [l.batch, l.out_c, l.out_h, l.out_w])
        l.artifacts = [this.webgl.programManager.build(l.glProg)];
        l.createRunData = WebGLUpsample.createRunData
      }
      else if (l.type.toUpperCase() == 'ROUTE') {
        l.textures = []
        if (l.input_layers.length == 1) l.textures.push({ groups: l.groups, TextureID: "t"+l.input_layers[0].index, shape: [l.input_layers[0].batch, l.out_c, l.input_layers[0].h, l.input_layers[0].w] })
        else for (let i = 0; i < l.input_layers.length; ++i) l.textures.push({ groups: l.groups, TextureID: "t"+l.input_layers[i].index, shape: [l.input_layers[i].batch, l.input_layers[i].c, l.input_layers[i].h, l.input_layers[i].w] })
        l.glProg = WebGLRoute.createProgramInfo(this.webgl, l.textures, [l.batch, l.out_c, l.out_h, l.out_w], 1)
        l.artifacts = [this.webgl.programManager.build(l.glProg)];
        l.createRunData = WebGLRoute.createRunData
      }
      else if (l.type.toUpperCase() == 'CONNECTED') {
        l.textures = [{TextureID: "t1", shape: [1000, 1], output: K.weights }, { TextureID: "t2w", shape: [1, 1000] }]
        l.glProg = WebGLMatMul.createProgramInfo(this.webgl, l)
        l.artifacts = [this.webgl.programManager.build(l.glProg)];
        l.createRunData = WebGLMatMul.createRunData
      }
      else if (l.type.toUpperCase() == 'SHORTCUT' || l.type.toUpperCase() == 'SAM') {
        l.textures = [{ TextureID: "t" + (l.index - 1), shape: [l.batch, l.c, l.h, l.w] }]
        l.glProg = WebGLSum.createProgramInfo(this.webgl, l)
        l.artifacts = [this.webgl.programManager.build(glProg)];
        l.createRunData = WebGLSum.createRunData
      }
    }
  }
  async start(img) {
    this.layers[0].output = ImageProcess.resize_image(img, this.layers[0].w, this.layers[0].h, 768, 576, 3)
    for (let i = 1; i < this.layers.length; ++i)await this.layers[i].forward(this.layers)
  }
  run(img) {
    this.layers[0].output = ImageProcess.resize_image(img, this.layers[0].w, this.layers[0].h, 768, 576, 3)
    this.layers[1].textures[0].output = this.layers[0].output
    console.time()
    for (let i = 1; i < this.layers.length; ++i) {
      if (this.layers[i].type.toUpperCase() == 'YOLO') continue
      if (!this.first) this.layers[i].runData = this.layers[i].createRunData(this.webgl);
      for (let j = 0; j < this.layers[i].artifacts.length; j++) {
        this.webgl.programManager.run(this.layers[i].artifacts[j], this.layers[i].runData[j]);
        this.layers[i].output = this.layers[i].runData[j].outputTextureData.gldata()
      }
    }
    console.timeEnd()
    this.first = true
  }
}
