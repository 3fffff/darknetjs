import {WebGLContext} from "./webgl-context.js"
import {ProgramManager} from "./program-manager.js"

export class WebGL {
  static cache = {}
  constructor(context) {
    this.context = context;
    try {
      this.glContext = createWebGLContext(this.context);
      if (typeof this.textureCacheMode !== 'string') this.textureCacheMode = 'full';
      console.log(`Created WebGLContext: ${JSON.stringify(this.glContext)}`);
    }
    catch (e) {
      console.log(`Unable to initialize WebGL. ${e}`);
    }
    this.textureDataCache = new Map();
    this.programManager = new ProgramManager(this.glContext);
  }
  getTextureData(tensorId) {
    return this.textureDataCache.get(tensorId);
  }
  setTextureData(tensorId, textureData) {
    //console.log('Storing Texture data in cache');
    this.textureDataCache.set(tensorId, textureData);
  }
  dispose() {
    this.programManager.dispose();
    this.clearActiveTextures();
    this.textureDataCache.forEach(td => this.releaseTexture(td, true));
    this.textureDataCache = new Map();
  }
  initFirstTexture(textureData, data) {
    const textureDataType = 'float';
    const encoder = this.glContext.getEncoder(textureDataType, 1, 1);
    console.log(`Init first image texture of size ${textureData.width}x${textureData.height}`);
    this.glContext.updateTexture(textureData.texture, textureData.width, textureData.height, encoder, this.toTextureData(data));
  }
  /**
   * Create a TextureData object from a tensor.
   * Usage = Encoder.Usage.UploadOnly.
   * If a related texture data is found in cache, returns it;
   * Otherwise:
   *   Creates a new texture layout if not provided;
   *   Creates WebGLTexture with the layout;
   *   Upload tensor data to the texture;
   *   Creates a texture data object associated with the given tensor.
   * @param tensor the tensor with data to upload
   */
  getOrCreateTextureData(tensor, layout,dim) {
    let td = this.getTextureData(tensor.TextureID);
    if (!td) {
      console.log(`Creating new TextureData for layer: [${tensor.TextureID}]`);
      if (!layout) layout = this.createTextureLayoutFromShape(dim);
      // graph inputs or initializers
      td = this.createTextureData(layout, 'float', tensor.TextureID, tensor.output, tensor, 1);
    }
    //else console.log(`Retrieving TextureData from cache: [${tensor.TextureID}]`);
    return td;
  }
  /**
   * Create a TextureData object from the given data type and texture layout.
   * Usage = Encoder.Usage.Default.
   * @param dataType the tensor data type
   */
  createTextureDataFromLayout(layout, dataType,TextureID) {
    return this.createTextureData(layout, dataType,TextureID);
  }
  /**
   * Create a TextureData object using the given data and bind to the given tensor.
   * Usage = Encoder.Usage.UploadOnly.
   * NOTE: this function is a hack for Conv implementation. should remove this function, after rewriting Conv
   * implementation by Graph.Transformer
   * @param dataType the tensor data type
   * @param data the actual data to upload
   * @param tensor the tensor to bind. tensor's data is ignored.
   */
  createTextureDataFromLayoutBindTensor(layout, dataType, data, tensor) {
    return this.createTextureData(layout, dataType, tensor.TextureID, data, tensor, 1 /* UploadOnly */);
  }
  createTextureData(layout, dataType = 'float32', TextureID, data, usage) {
    //console.log(`Creating TextureData: layout:[${JSON.stringify(layout)}]`);
    const texture = this.createTextureFromLayout(dataType, layout, data, usage);
    return this.createTextureDataFromTexture(layout, dataType, texture, TextureID);
  }
  /**
   * Create a TextureData object, using the given texture.
   * This function does not create new texture. Usually used in scenarios using texture sharing. (eg. Reshape)
   * @param dataType the tensor data type
   * @param texture the WebGLTexture object to share
   * @param tensorId the tensor ID of the shared tensor data
   */
  createSharedTextureData(layout, dataType, texture, tensorId) {
    return this.createTextureDataFromTexture(layout, dataType, texture, tensorId);
  }
  createTextureDataFromTexture(layout, dataType, texture, tensorId) {
    const textureData = Object.assign(Object.assign({}, layout), {
      gldata: () => {
        return this.readTexture(textureData);
      }, texture
    });
    this.setTextureData(tensorId, textureData);
    return textureData;
  }
  /**
   * Create a TextureLayout object from a tensor. If a related texture data is found, returns the cached texture layout.
   */
  getOrCreateTextureLayout(TextureID,dims, channels = 1, unpackedShape) {
    const td = this.getTextureData(TextureID);
    if (td) return td;
    return this.createTextureLayoutFromShape(channels === 1 ? dims : getPackedShape(dims), channels, unpackedShape);
  }
  /**
   * Create a TextureLayout object from shape.
   */
  createTextureLayoutFromShape(shape, channels = 1, unpackedShape, prefs) {
    const [width, height] = this.computeTextureWH(shape, prefs);
    let inferredDims = shape;
    if (shape.length === 0) {
      inferredDims = [1];
    }
    if (channels === 1) {
      // unpackedShape will take `shape` and not `inferredDims` so as to create a scalar Tensor if need be
      unpackedShape = shape;
    }
    else if (!unpackedShape) {
      throw new Error('Unpacked shape is needed when using channels > 1');
    }
    return {
      width,
      height,
      channels: channels ? channels : 1,
      shape: inferredDims,
      strides: WebGL.computeStrides(inferredDims),
      unpackedShape
    };
  }
  readTexture(textureData) {
    if (!this.glContext.isFloat32DownloadSupported) {
      const op = new WebGLUint8Encode();
      const uint8TD = op.runInternal(this, textureData);
      return this.readUint8TextureAsFloat(uint8TD);
    }
    return this.readTextureAsFloat(textureData, 'float32', textureData.channels);
  }

  toTextureData(data) {
    if (!data) return undefined;
    return (data instanceof Float32Array) ? data : new Float32Array(data);
  }
  createTextureFromLayout(dataType, layout, data, usage) {
    const textureDataType = 'float';
    const encoder = this.glContext.getEncoder(textureDataType, layout.channels || 1, usage);
    //console.log('TextureManager', `Creating new texture of size ${layout.width}x${layout.height}`);
    const texture = this.glContext.allocateTexture(layout.width, layout.height, encoder, this.toTextureData( data));
    return texture;
  }
  readTextureAsFloat(td, dataType, channels) {
    if (!channels) channels = 1;
    const dataSize = td.shape.reduce((a, b) => a * b) * channels;
    return this.glContext.readTexture(td.texture, td.width, td.height, dataSize, dataType, channels);
  }
  readUint8TextureAsFloat(td) {
    const dataSize = td.shape.reduce((a, b) => a * b);
    const data = this.glContext.readTexture(td.texture, td.width, td.height, dataSize * 4, 'byte', 4);
    return new Float32Array(data.buffer, data.byteOffset, dataSize);
  }
  releaseTexture(textureData, deleteTexture) {
    if (deleteTexture) {
      console.log('TextureManager', `Deleting texture of size ${textureData.width}x${textureData.height}`);
      this.glContext.deleteTexture(textureData.texture);
    }
  }
  clearActiveTextures() {
    this.glContext.clearActiveTextures();
  }

  /**
   * This strategy try to find the minimal max(W,H) that fulfills (W * H == totalSize)
   */
  computeTextureWH(shape, prefs) {
    // scalar tensor
    if (shape.length === 0) return [1, 1];
    const maxTextureSize = this.glContext.maxTextureSize;
    if (prefs) {
      // check to see if dims fit
      const wsize = prefs.breakAxis >= shape.length ? 1 : shape.slice(prefs.breakAxis).reduce((a, b) => a * b);
      const hsize = prefs.breakAxis <= 0 ? 1 : shape.slice(0, prefs.breakAxis).reduce((a, b) => a * b);
      if (wsize > maxTextureSize || hsize > maxTextureSize) {
        // ignore preferences
        // continue with default layout
        console.log('TextureLayout', `Given width/height preferences were unattainable: shape:${shape}, breakAxis:${prefs.breakAxis}`);
      }
      else {
        return [wsize, hsize];
      }
    }
    const totalSize = shape.reduce((a, b) => a * b);
    let width = Math.floor(Math.sqrt(totalSize));
    for (; width < maxTextureSize && width < totalSize; width++)
      if (totalSize % width === 0)
        break;
    if (width >= maxTextureSize || totalSize % width !== 0) {
      throw new Error(`The given dimensions are outside this GPU\'s boundaries: ${shape}`);
    }
    return [width, totalSize / width];
  }
  static computeStrides(dims) {
    const rank = dims.length;
    if (rank === 0) {
      return [];
    } else if (rank === 1) {
      return [1];
    }
    const strides = new Array(rank);
    strides[rank - 1] = 1;
    strides[rank - 2] = dims[rank - 1];
    for (let i = rank - 3; i >= 0; --i) {
      strides[i] = strides[i + 1] * dims[i + 1];
    }
    return strides;
  }
}
/**
 * Given a non RGBA shape calculate the R version
 * It is assumed that the dimensions are multiples of given channels
 * NOTE: it is always the last dim that gets packed.
 * @param unpackedShape original shape to create a packed version from
 */
function getPackedShape(unpackedShape) {
  const len = unpackedShape.length;
  return unpackedShape.slice(0, len - 1).concat(unpackedShape[len - 1] / 4);
}
/**
 * This factory function creates proper WebGLRenderingContext based on
 * the current browsers capabilities
 * The order is from higher/most recent versions to most basic
 */
function createWebGLContext(contextId) {
  let context;
  if ((!contextId || contextId === 'webgl2') && 'webgl2' in WebGL.cache) {
    context = WebGL.cache.webgl2;
  }
  else if ((!contextId || contextId === 'webgl') && 'webgl' in WebGL.cache) {
    context = WebGL.cache.webgl;
  }
  context = context || createNewWebGLContext(contextId);
  contextId = contextId || context.version === 1 ? 'webgl' : 'webgl2';
  const gl = context.gl;
  WebGL.cache[contextId] = context;
  if (gl.isContextLost()) {
    delete WebGL.cache[contextId];
    return createWebGLContext(contextId);
  }
  gl.disable(gl.DEPTH_TEST);
  gl.disable(gl.STENCIL_TEST);
  gl.disable(gl.BLEND);
  gl.disable(gl.DITHER);
  gl.disable(gl.POLYGON_OFFSET_FILL);
  gl.disable(gl.SAMPLE_COVERAGE);
  gl.enable(gl.SCISSOR_TEST);
  gl.enable(gl.CULL_FACE);
  gl.cullFace(gl.BACK);
  return context;
}
function createNewWebGLContext(contextId) {
  const canvas = createCanvas();
  const contextAttributes = {
    alpha: false,
    depth: false,
    antialias: false,
    stencil: false,
    preserveDrawingBuffer: false,
    premultipliedAlpha: false,
    failIfMajorPerformanceCaveat: false
  };
  let gl;
  const ca = contextAttributes;
  if (!contextId || contextId === 'webgl2') {
    gl = canvas.getContext('webgl2', ca);
    if (gl) {
      try {
        return new WebGLContext(gl, 2);
      }
      catch (err) {
        console.log('GlContextFactory', `failed to create WebGLContext using contextId 'webgl2'. Error: ${err}`);
      }
    }
  }
  if (!contextId || contextId === 'webgl') {
    gl = canvas.getContext('webgl', ca) || canvas.getContext('experimental-webgl', ca);
    if (gl) {
      try {
        return new WebGLContext(gl, 1);
      }
      catch (err) {
        console.log('GlContextFactory', `failed to create WebGLContext using contextId 'webgl' or 'experimental-webgl'. Error: ${err}`);
      }
    }
  }
  throw new Error('WebGL is not supported');
}
function createCanvas() {
  const canvas = document.createElement('canvas');
  canvas.width = 1;
  canvas.height = 1;
  return canvas;
}