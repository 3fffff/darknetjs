class WasmBinding {
  static instance;
  constructor() {
    this.ptr8 = 0;
    this.numBytesAllocated = 0;
  }

  ccall(functionName, ...params) {
    /* if (!initializedWasm) {
         throw new Error(`wasm not initializedWasm. please ensure 'init()' is called.`);
     }*/
    const offset = [];
    const size = WasmBinding.calculateOffsets(offset, params);
    if (size > this.numBytesAllocated) this.expandMemory(size);

    WasmBinding.ccallSerialize(Module.HEAPU8.subarray(this.ptr8, this.ptr8 + size), offset, params);
    this.func(functionName, this.ptr8);
    WasmBinding.ccallDeserialize(Module.HEAPU8.subarray(this.ptr8, this.ptr8 + size), offset, params);
  }
  // raw ccall method  without invoking ccallSerialize() and ccallDeserialize()
  // user by ccallRemote() in the web-worker
  ccallRaw(functionName, data) {
    /*if (!initializedWasm) {
        throw new Error(`wasm not initializedWasm. please ensure 'init()' is called.`);
    }*/
    const size = data.byteLength;
    if (size > this.numBytesAllocated) this.expandMemory(size);
    // copy input memory (data) to WASM heap
    Module.HEAPU8.subarray(this.ptr8, this.ptr8 + size).set(data);
    this.func(functionName, this.ptr8);
    // copy Wasm heap to output memory (data)
    data.set(Module.HEAPU8.subarray(this.ptr8, this.ptr8 + size));
  }
  func(functionName, ptr8) {
    const func = Module[functionName];
    func(ptr8);
  }
  static calculateOffsets(offset, params) {
    // calculate size and offset
    let size = 4 + 4 * params.length;
    for (let i = 0; i < params.length; i++) {
      const param = params[i];
      const paramData = param[0];
      const paramType = param[1];
      const paramPass = param[2];
      let len = 0;
      switch (paramType) {
        case 'bool':
        case 'int32':
        case 'float32':
          len = 4;
          break;
        case 'float64':
          len = 8;
          break;
        case 'boolptr':
          if (!paramData) {
            // deal with nullptr
            offset.push(0);
            continue;
          }
          else if (Array.isArray(paramData) || ArrayBuffer.isView(paramData)) {
            len = 4 * Math.ceil(paramData.length / 4);
          }
          else {
            throw new Error(`boolptr requires boolean array or Uint8Array`);
          }
          break;
        case 'int32ptr':
        case 'float32ptr':
          if (!paramData) {
            // deal with nullptr
            offset.push(0);
            continue;
          }
          else if (Array.isArray(paramData)) {
            if (paramPass === 'inout' || paramPass === 'out') {
              throw new TypeError(`inout/out parameters must be ArrayBufferView for ptr types.`);
            }
            len = paramData.length * 4;
          }
          else if (ArrayBuffer.isView(paramData)) {
            len = paramData.byteLength;
          }
          else {
            throw new TypeError(`unsupported data type in 'ccall()'`);
          }
          break;
        default:
          throw new Error(`not supported parameter type: ${paramType}`);
      }
      offset.push(size);
      size += len;
    }
    return size;
  }
  // tranfer data parameters (in/inout) to emscripten heap for ccall()
  static ccallSerialize(heapU8, offset, params) {
    const heap32 = new Int32Array(heapU8.buffer, heapU8.byteOffset);
    const heapU32 = new Uint32Array(heapU8.buffer, heapU8.byteOffset);
    const heapF32 = new Float32Array(heapU8.buffer, heapU8.byteOffset);
    heapU32[0] = params.length;
    for (let i = 0; i < params.length; i++) {
      const param = params[i];
      const paramData = param[0];
      const paramType = param[1];
      // const paramPass = param[2];
      const offset8 = offset[i];
      const offset32 = offset8 >> 2;
      heapU32[i + 1] = offset8;
      if (/*paramPass === 'out' ||*/ offset8 === 0) continue;
      switch (paramType) {
        case 'bool':
          heapU8[offset8] = paramData === true ? 1 : 0;
          break;
        case 'int32':
          heap32[offset32] = paramData;
          break;
        case 'float32':
          heapF32[offset32] = paramData;
          break;
        case 'boolptr':
          const boolArray = paramData;
          // This will work for both Uint8Array as well as ReadonlyArray<boolean>
          heapU8.subarray(offset8, offset8 + boolArray.length).set(paramData);
          break;
        case 'int32ptr':
          const int32Array = paramData;
          heap32.subarray(offset32, offset32 + int32Array.length).set(int32Array);
          break;
        case 'float32ptr':
          const float32Array = paramData;
          heapF32.subarray(offset32, offset32 + float32Array.length).set(float32Array);
          break;
        default:
          throw new Error(`not supported parameter type: ${paramType}`);
      }
    }
  }
  // retrieve data parameters (in/inout) from emscripten heap after ccall()
  static ccallDeserialize(buffer, offset, params) {
    const heapF32 = new Float32Array(buffer.buffer, buffer.byteOffset);
    const heapU8 = new Uint8Array(buffer.buffer, buffer.byteOffset);
    for (let i = 0; i < params.length; i++) {
      const param = params[i];
      const paramData = param[0];
      const paramType = param[1];
      const paramPass = param[2];
      const offset8 = offset[i];
      // const offset16 = offset8 >> 1;
      const offset32 = offset8 >> 2;
      // const offset64 = offset8 >> 3;
      if (paramPass !== 'out' && paramPass !== 'inout') {
        continue;
      }
      switch (paramType) {
        case 'float32ptr':
          const float32Array = paramData;
          float32Array.set(heapF32.subarray(offset32, offset32 + float32Array.length));
          break;
        case 'boolptr':
          const boolArray = paramData;
          boolArray.set(heapU8.subarray(offset8, offset8 + boolArray.length));
          break;
        default:
          throw new Error(`not supported parameter type: ${paramType}`);
      }
    }
  }
  // function for defining memory allocation strategy
  expandMemory(minBytesRequired) {
    // free already held memory if applicable
    if (this.ptr8 !== 0) Module._free(this.ptr8);
    // current simplistic strategy is to allocate 2 times the minimum bytes requested
    this.numBytesAllocated = 2 * minBytesRequired;
    this.ptr8 = Module._malloc(this.numBytesAllocated);
    if (this.ptr8 === 0) throw new Error('Unable to allocate requested amount of memory. Failing.');
  }
  dispose() {
    /*if (!initializedWasm) {
        throw new Error(`wasm not initializedWasm. please ensure 'init()' is called.`);
    }*/
    if (this.ptr8 !== 0) {
      Module._free(this.ptr8);
      this.numBytesAllocated = 0;
    }
  }
  static getInstance() {
    if (!WasmBinding.instance) WasmBinding.instance = new WasmBinding();
    return WasmBinding.instance;
  }
  ccallRemote(workerId, functionName, ...params) {
    /*  if (!initializedWorker) {
          throw new Error(`wasm not initializedWorker. please ensure 'init()' is called.`);
      }*/
    if (workerId < 0 /*|| workerId >= WasmBinding.WORKER_NUMBER*/) {
      throw new Error(`invalid worker ID ${workerId}. should be in range [0, ${WasmBinding.WORKER_NUMBER})`);
    }
    const offset = [];
    const size = WasmBinding.calculateOffsets(offset, params);
    const buffer = new ArrayBuffer(size);
    WasmBinding.ccallSerialize(new Uint8Array(buffer), offset, params);
    workers[workerId].postMessage({ type: 'ccall', func: functionName, buffer }, [buffer]);
    return new Promise((resolve, reject) => {
      completeCallbacks[workerId].push((buffer, perf) => {
        WasmBinding.ccallDeserialize(new Uint8Array(buffer), offset, params);
        resolve(perf);
      });
    });
  }
}