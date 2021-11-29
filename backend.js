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
    /*const routeInp = new Float32Array(346112)
    for (let i = 0; i < routeInp.length; i++)routeInp[i] = Math.random() > 0.5 ? -Math.random() : Math.random()
    const route = { batch: 1,  group_id: 1, groups: 2, index: 4, glInputs: [{ index: 1, groups:1,TextureID: "t1", shape: [1, 32, 104, 104], output: routeInp }], input_layers: Array, input_sizes: Array(692224), inputs: 346112, out_c: 32, out_h: 104, out_w: 104, output: new Float32Array(346112), outputs: 346112, type: "ROUTE" }
    route.glProg = WebGLRoute.createProgramInfo(this.webgl, route.glInputs, [route.batch, route.out_c, route.out_h, route.out_w],1)
    route.glData = WebGLRoute.createRunData
    route.artifact = this.webgl.programManager.build(route.glProg);
    this.webgl.programManager.setArtifact(route.TextureID, route.artifact);
    route.runData = route.glData(this.webgl, route.glInputs);
    this.webgl.programManager.run(route.artifact, route.runData);
    console.log(routeInp)
    console.log(route.runData.outputTextureData.gldata())*/
    /*const convInp = new Float32Array(1 * 3 * 416 * 416)
    for (let i = 0; i < convInp.length; i++)convInp[i] = Math.random() > 0.5 ? -Math.random() : Math.random()
    const kernelInp = new Float32Array(864)
    for (let i = 0; i < kernelInp.length; i++)kernelInp[i] = Math.random() > 0.5 ? -Math.random() : Math.random()
    const biasInp = new Float32Array(32)
    for (let i = 0; i < biasInp.length; i++)biasInp[i] = Math.random() > 0.5 ? -Math.random() : Math.random()
    const CONV = { index: 2, type: "CONVOLUTIONAL", TextureID: "t2", batch: 1, biases: biasInp, weights: kernelInp, pad: 1, size: 3, stride_x: 2, stride_y: 2, c: 3, h: 416, w: 416, out_c: 32, out_h: 208, out_w: 208, output: new Float32Array(1 * 32 * 208 * 208) }
    CONV.glInputs = [{ index: 1, TextureID: "t" + (1), pad: 1, size: 2, stride_x: 2, stride_y: 2, shape: [1, 3, 416, 416], output: convInp }, { batch: 1, filters: 32, size: 3, weights: kernelInp, dilation: 1, TextureID: 'w2', groups:1,shape: [3, 32, 3, 3], pad: 1, stride_x: 2, stride_y: 2 }, { output: biasInp, TextureID: 'bias2', shape: [32] }]
    CONV.glProg = WebGLConv.createProgramInfos(this.webgl, CONV.glInputs, [CONV.batch, CONV.out_c, CONV.out_h, CONV.out_w])
    CONV.glData = WebGLConv.createRunDatas
    CONV.artifacts = []
    CONV.artifacts.push(this.webgl.programManager.build(CONV.glProg[0]));
    CONV.artifacts.push(this.webgl.programManager.build(CONV.glProg[1]));
    this.webgl.programManager.setArtifact("t1", CONV.artifacts[0]);
    this.webgl.programManager.setArtifact(CONV.TextureID, CONV.artifacts[1]);
    CONV.runData = CONV.glData(this.webgl, CONV.glInputs);
    this.webgl.programManager.run(CONV.artifacts[0], CONV.runData[0]);
    this.webgl.programManager.run(CONV.artifacts[1], CONV.runData[1]);
    console.log(CONV.runData[1].outputTextureData.gldata())*/
    //const l0 = { index: 0, TextureID: "t0", b: 1, c: 16, h: 304, w: 304, n: 16, scale: new Float32Array(16), mean: new Float32Array(16), variance: new Float32Array(16), out_c: 16, out_h: 304, out_w: 304, output: Array(16 * 304 * 304) }
    //const l1 = { index: 1, TextureID: "t1", b: 1, c: 16, h: 304, w: 304, n: 16, scale: new Float32Array(16), mean: new Float32Array(16), variance: new Float32Array(16), out_c: 16, out_h: 304, out_w: 304, output: Array(16 * 304 * 304) }
    //const l2 = { index: 2, type: "SHORTCUT", activation: "RELU",glInputs:[{  TextureID: "t"+(l.index-1),shape:[l.b, l.c, l.h, l.w]}, { TextureID: "t"+l.indexs,shape:[l.b, l.c, l.h, l.w] }], TextureID:"t2",b: 1, c: 16, h: 304, w: 304, n: 16, indexs: 0, out_c: 16, out_h: 304, out_w: 304, output: Array(16 * 304 * 304) }
    //const l3 = { index: 3, type: "SHORTCUT", activation: "RELU", TextureID:"t3", glInputs:[{TextureID: "t2",shape:[l.b, l.c, l.h, l.w]}], b: 1, c: 16, h: 304, w: 304, n: 16, scale: new Float32Array(16), mean: new Float32Array(16), variance: new Float32Array(16), out_c: 16, out_h: 304, out_w: 304, output: Array(16 * 304 * 304) }
    //const BN = { index:2, type: "BATCHNORM",TextureID:"t2",b: 1,glInputs:[{ index: l.index, TextureID: "t"+(l.index-1),shape:[1000,1]}, { index: l.index, TextureID: "t"+l.index+"w",shape:[1,1000] }], weights:new Float32Array(1000),c: 1000, h: 1, w: 1, out_c: 1000000, out_h: 1, out_w: 1, output: new Float32Array(1 * 1 * 1000000) }
    //const K = { index:2, type: "MATMUL",TextureID:"t2",b: 1,glInputs:[{ index: l.index, TextureID: "t"+(l.index-1),shape:[1000,1]}, { index: l.index, TextureID: "t"+l.index+"w",shape:[1,1000] }], weights:new Float32Array(1000),c: 1000, h: 1, w: 1, out_c: 1000000, out_h: 1, out_w: 1, output: new Float32Array(1 * 1 * 1000000) }
    /*const maxpoolInp = new Float32Array(1 * 128 * 104 * 104)
    for (let i = 0; i < maxpoolInp.length; i++)maxpoolInp[i] = Math.random() > 0.5 ? -Math.random() : Math.random()
    const MAXPOOL = { index: 2, type: "MAXPOOL", TextureID: "t2", batch: 1, pad: 1, size: 2, stride_x: 2, stride_y: 2, glInputs: { index: 1, TextureID: "t" + (1), pad: 1, size: 2, stride_x: 2, stride_y: 2, shape: [1, 128, 104, 104], output: maxpoolInp }, c: 128, h: 104, w: 104, out_c: 128, out_h: 52, out_w: 52, output: new Float32Array(1 * 128 * 52 * 52) }
    MAXPOOL.glProg = WebGLPool.createProgramInfo(this.webgl, MAXPOOL.glInputs, [MAXPOOL.batch, MAXPOOL.out_c, MAXPOOL.out_h, MAXPOOL.out_w])
    MAXPOOL.glData = WebGLPool.createRunData
    MAXPOOL.artifact = this.webgl.programManager.build(MAXPOOL.glProg);
    this.webgl.programManager.setArtifact(MAXPOOL.TextureID, MAXPOOL.artifact);
    MAXPOOL.runData = MAXPOOL.glData(this.webgl, MAXPOOL.glInputs);
    this.webgl.programManager.run(MAXPOOL.artifact, MAXPOOL.runData);
    console.log(MAXPOOL.runData.outputTextureData.gldata())*/
    /*const upInp = new Float32Array(1 * 128 * 104 * 104)
    for (let i = 0; i < upInp.length; i++)upInp[i] = Math.random() > 0.5 ? -Math.random() : Math.random()
    const UPSAMPLE = { index: 2, type: "UPSAMPLE", TextureID: "t2",scale:1,stride:2, batch: 1, pad: 1, glInputs: { index: 1, TextureID: "t" + (1),scale:1,stride:2, shape: [1, 128, 104, 104], output: upInp }, c: 128, h: 104, w: 104, out_c: 128, out_h: 208, out_w: 208, output: new Float32Array(1 * 128 * 208 * 208) }
    UPSAMPLE.glProg = WebGLUpsample.createProgramInfo(this.webgl, UPSAMPLE.glInputs, [UPSAMPLE.batch, UPSAMPLE.out_c, UPSAMPLE.out_h, UPSAMPLE.out_w])
    UPSAMPLE.glData = WebGLUpsample.createRunData
    UPSAMPLE.artifact = this.webgl.programManager.build(UPSAMPLE.glProg);
    this.webgl.programManager.setArtifact(UPSAMPLE.TextureID, UPSAMPLE.artifact);
    UPSAMPLE.runData = UPSAMPLE.glData(this.webgl, UPSAMPLE.glInputs);
    this.webgl.programManager.run(UPSAMPLE.artifact, UPSAMPLE.runData);
    console.log(UPSAMPLE.runData.outputTextureData.gldata())*/
    /*K.glProg = WebGLMatMul.createProgramInfo(this.webgl, K)
    K.glData = WebGLMatMul.createRunData
    const inputs = [{ index: 2, TextureID: "t1",shape:[1000,1],output:K.weights}, { index: 2, TextureID: "t2w",shape:[1,1000],output:new Float32Array(1000) }]
    l3.glData = WebGLActivation.createRunData
    K.artifact = this.webgl.programManager.build(K.glProg);
    for (let i = 0; i < l0.output.length; i++)inputs[0].output[i] = Math.random() > 0.5 ? -Math.random() : Math.random()
    for (let i = 0; i < l1.output.length; i++)inputs[1].output[i] = Math.random() > 0.5 ? -Math.random() : Math.random()
    K.runData = K.glData(this.webgl, inputs);
    console.log(K)
    this.webgl.programManager.run(K.artifact, K.runData);
    console.log(K.runData.outputTextureData.gldata())
    const output = new Float32Array(1000000)
    Forward.matmul(inputs[0].output,inputs[1].output, output,1000, 1000, 1);
    console.log(output)*/
    /* const lr = [l2, l3]
     for (let i = 0; i < l0.output.length; i++)l0.output[i] = Math.random() > 0.5 ? -Math.random() : Math.random()
     for (let i = 0; i < l1.output.length; i++)l1.output[i] = Math.random() > 0.5 ? -Math.random() : Math.random()
     const inputs = [{ index: l2.index, TextureID:"t2",TextureID: "t1", b: l1.b, c: l1.c, h: l1.h, w: l1.w, output: l1.output }, { index: l2.index, TextureID: "t0", b: l0.b, c: l0.c, h: l0.h, w: l0.w, output: l0.output }]
     l2.glProg = WebGLSum.createProgramInfo(this.webgl, l2)
     l2.glData = WebGLSum.createRunData
     console.log(l2.glProg)
     l2.artifact = this.webgl.programManager.build(l2.glProg);
     l2.runData = l2.glData(this.webgl, inputs);
     l3.glProg = WebGLActivation.createProgramInfo(this.webgl, l3)
     l3.glData = WebGLActivation.createRunData
     console.log(l3.glProg)
     l3.artifact = this.webgl.programManager.build(l3.glProg);
     this.webgl.programManager.setArtifact(lr[0].TextureID, lr[0].artifact);
     this.webgl.programManager.setArtifact(lr[1].TextureID, lr[1].artifact);
     lr[1].runData = lr[1].glData(this.webgl, lr[0].runData.outputTextureData.gldata());
     console.log(inputs[0].output)
     console.log(inputs[1].output)
     //  for (let i = 0; i < lr.length; i++) {
     console.time()
     this.webgl.programManager.run(lr[0].artifact, lr[0].runData);
     console.log(lr[0].runData.outputTextureData.gldata())
     this.webgl.programManager.run(lr[1].artifact, lr[1].runData);
     console.log(lr[1].runData.outputTextureData.gldata())
     console.timeEnd()*/
    // }
    ////////////////////////////////////////////////
  }
  async start(img) {
    this.layers[0].output = ImageProcess.resize_image(img, this.layers[0].w, this.layers[0].h, 768, 576, 3)
    for (let i = 1; i < this.layers.length; ++i)await this.layers[i].forward(this.layers)
  }
  run(img) {
    this.layers[0].output = ImageProcess.resize_image(img, this.layers[0].w, this.layers[0].h, 768, 576, 3)
    for (let i = 1; i < this.layers.length; ++i) this.webgl.programManager.run(this.layers[i].artifact, this.layers[i].glData);
  }
}
