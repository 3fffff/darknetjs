class Backend {
  constructor(layers) {
    this.layers = layers;
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
        const textures = [{ index: l.index, activation: l.activation, TextureID: "t" + (l.index - 1), activation: l.activation, pad: l.pad, size: l.size, shape: [l.batch, l.c, l.h, l.w] },
        { batch: l.batch, filters: l.filters, size: l.size, output: l.weights, dilation: l.dilation, TextureID: 'w' + l.index, groups: l.groups, shape: [l.filters, l.c, l.size, l.size], pad: l.pad, stride_x: l.stride_x, stride_y: l.stride_y },
        { output: l.biases, TextureID: 'bias' + l.index, shape: [l.filters] }]
        l.artifacts = []
        if (l.groups == 1) {
          const glProg = WebGLConv.createProgramInfos(this.webgl, textures, [l.batch, l.out_c, l.out_h, l.out_w], l.activation)
          l.artifacts.push(this.webgl.programManager.build(glProg[0]));
          l.artifacts.push(this.webgl.programManager.build(glProg[1]));
          if (glProg[2]) l.artifacts.push(this.webgl.programManager.build(glProg[2]));
          l.runData = WebGLConv.createRunDatas(this.webgl, textures, glProg, l.index, l.activation)
        }
        else {
          const glProg = WebGLGroupConv.createProgramInfos(this.webgl, textures, [l.batch, l.out_c, l.out_h, l.out_w], l.activation)
          l.artifacts.push(this.webgl.programManager.build(glProg[0]));
          if (glProg[1]) l.artifacts.push(this.webgl.programManager.build(glProg[1]));
          l.runData = WebGLGroupConv.createRunDatas(this.webgl, textures, glProg, l.index, l.activation)
        }
      }
      else if (l.type.toUpperCase() == 'MAXPOOL' || l.type.toUpperCase() == 'LOCALAVG' || l.type.toUpperCase() == 'AVGPOOL') {
        const textures = [{ TextureID: "t" + (l.index - 1), pad: l.pad, size: l.size, stride_x: l.stride_x, stride_y: l.stride_x, shape: [l.batch, l.c, l.h, l.w] }]
        const glProg = WebGLPool.createProgramInfo(this.webgl, textures[0], [l.batch, l.out_c, l.out_h, l.out_w], l.type)
        l.artifacts = [this.webgl.programManager.build(glProg)]
        l.runData = WebGLPool.createRunData(this.webgl, textures, glProg, l.index)
      }
      else if (l.type.toUpperCase() == 'UPSAMPLE') {
        const textures = [{ TextureID: "t" + (l.index - 1), scale: 1, stride: l.stride, shape: [l.batch, l.c, l.h, l.w] }]
        const glProg = WebGLUpsample.createProgramInfo(this.webgl, textures[0], [l.batch, l.out_c, l.out_h, l.out_w])
        l.artifacts = [this.webgl.programManager.build(glProg)];
        l.runData = WebGLUpsample.createRunData(this.webgl, textures, glProg, l.index)
      }
      else if (l.type.toUpperCase() == 'ROUTE') {
        const textures = []
        let glProg;
        if (l.input_layers.length == 1) {
          textures.push({ groups: 0, TextureID: "t" + l.input_layers[0].index, shape: [l.input_layers[0].batch, l.out_c, l.input_layers[0].h, l.input_layers[0].w] })
          glProg = WebGLRoute.createSplitProgramInfo(this.webgl, textures, [l.batch, l.out_c, l.out_h, l.out_w], l.group_id * l.out_c)
        } else {
          for (let i = 0; i < l.input_layers.length; ++i) textures.push({ groups: l.groups, TextureID: "t" + l.input_layers[i].index, shape: [l.input_layers[i].batch, l.input_layers[i].out_c, l.input_layers[i].out_h, l.input_layers[i].out_w] })
          glProg = WebGLRoute.createProgramInfo(this.webgl, textures, [l.batch, l.out_c, l.out_h, l.out_w])
        }
        l.artifacts = [this.webgl.programManager.build(glProg)];
        l.runData = WebGLRoute.createRunData(this.webgl, textures, glProg, l.index)
      }
      else if (l.type.toUpperCase() == 'CONNECTED') {
        const textures = [{ TextureID: "tw" + l.index, shape: [l.weights.length, 1], output: l.weights }, { TextureID: "t" + (l.index - 1), shape: [1, l.weights.length] }]
        const glProg = WebGLMatMul.createProgramInfo(this.webgl, l)
        l.artifacts = [this.webgl.programManager.build(glProg)];
        l.runData = WebGLMatMul.createRunData(this.webgl, textures, glProg, l.index)
      }
      else if (l.type.toUpperCase() == 'SHORTCUT' || l.type.toUpperCase() == 'SAM') {
        const textures = [{ TextureID: "t" + (l.index - 1), shape: [l.batch, l.c, l.h, l.w] }, { TextureID: "t" + l.indexs, shape: [l.batch, l.c, l.h, l.w] }]
        const glProg = WebGLSum.createProgramInfo(this.webgl, textures, [l.batch, l.out_c, l.out_h, l.out_w], l.type.toUpperCase())
        l.artifacts = [this.webgl.programManager.build(glProg)];
        l.runData = WebGLSum.createRunData(this.webgl, textures, glProg, l.index)
      }
      else if (l.type.toUpperCase() == 'SCALE_CHANNELS') {
        const textures = [{ TextureID: "t" + (l.index - 1), shape: [l.batch, l.c, l.h, l.w] }, { TextureID: "t" + l.indexs, shape: [l.batch, l.c, l.h, l.w] }]
        const glProg = WebGLSum.createScaleChannelsProgramInfo(this.webgl, textures, [l.batch, l.out_c, l.out_h, l.out_w], l.type.toUpperCase())
        l.artifacts = [this.webgl.programManager.build(glProg)];
        l.runData = WebGLSum.createRunData(this.webgl, textures, glProg, l.index)
      }
      else if (l.type.toUpperCase() == 'DROPOUT') {
        l.artifacts = this.layers[i - 1].artifacts
        l.runData = this.layers[i - 1].runData
      }
    }
  }
  async start(img, width, height, channels = 3) {
    this.layers[0].output = ImageProcess.resize_image(img, this.layers[0].w, this.layers[0].h, width, height, channels)
    //for(let i=0;i<this.layers[0].output.length;i++)this.layers[0].output[i] = 0
    for (let i = 1; i < this.layers.length; ++i)await this.layers[i].forward(this.layers)
  }
  run(img, width, height, channels = 3) {
    this.layers[0].output = ImageProcess.resize_image(img, this.layers[0].w, this.layers[0].h, width, height, channels)
    this.webgl.initFirstTexture(this.layers[1].runData[0].inputTextureDatas[0], this.layers[0].output)
    console.time()
    for (let i = 1; i < this.layers.length; ++i) {
      if (this.layers[i].type.toUpperCase() == 'YOLO') {
        const len = this.layers[i - 1].runData.length
        this.layers[i].output = this.layers[i - 1].runData[len - 1].outputTextureData.gldata()
        continue
      }
      for (let j = 0; j < this.layers[i].artifacts.length; j++) {
        this.webgl.programManager.run(this.layers[i].artifacts[j], this.layers[i].runData[j]);
        //this.layers[i].output = this.layers[i].runData[j].outputTextureData.gldata()
      }
    }
    console.timeEnd()
  }
}
