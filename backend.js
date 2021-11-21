class Backend {
  constructor(layers,wasm,webgl){
    this.wasm = wasm ? true : false
    this.webgl = webgl ? new WebGL("webgl2") : null
    this.layers = layers
    if(wasm)this.initWasm()
  }
  initWasm(){
    const wasmSupport = wasmcheck.support()
    const Simd = wasmcheck.feature.simd
    const thread = wasmcheck.feature.threads
    if(!wasmSupport)throw new Error("wasm is not supported")
    if(!Simd)throw new Error("simd is not supported")
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
  async start(img) {
    this.layers[0].output = ImageProcess.resize_image(img, this.layers[0].w, this.layers[0].h, 768, 576, 3)
    for (let i = 1; i < this.layers.length; ++i)await this.layers[i].forward(this.layers)
  }
  run(img) {
    this.layers[0].output = ImageProcess.resize_image(img, this.layers[0].w, this.layers[0].h, 768, 576, 3)
    for (let i = 1; i < this.layers.length; ++i) this.webgl.programManager.run(this.layers[i].artifact, this.layers[i].glData);
  }
}
