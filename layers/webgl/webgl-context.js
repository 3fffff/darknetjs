"use strict";
/**
 * Abstraction and wrapper around WebGLRenderingContext and its operations
 */
class WebGLContext {
  constructor(gl, version) {
    this.frameBufferBound = false;
    this.gl = gl;
    this.version = version;
    this.getExtensions();
    this.vertexbuffer = this.createVertexbuffer();
    this.framebuffer = this.createFramebuffer();
    this.queryVitalParameters();
  }
  allocateTexture(width, height, encoder, data) {
    const gl = this.gl;
    // create the texture
    const texture = gl.createTexture();
    // bind the texture so the following methods effect this texture.
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const buffer = data ? encoder.encode(data, width * height) : null;
    gl.texImage2D(gl.TEXTURE_2D, 0, // Level of detail.
      encoder.internalFormat, width, height, 0, // Always 0 in OpenGL ES.
      encoder.format, encoder.textureType, buffer);
    this.checkError();
    return texture;
  }
  updateTexture(texture, width, height, encoder, data) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    const buffer = encoder.encode(data, width * height);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, // level
      0, // xoffset
      0, // yoffset
      width, height, encoder.format, encoder.textureType, buffer);
    this.checkError();
  }
  attachFramebuffer(texture, width, height) {
    const gl = this.gl;
    // Make it the target for framebuffer operations - including rendering.
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0); // 0, we aren't using MIPMAPs
    this.checkError();
    gl.viewport(0, 0, width, height);
    gl.scissor(0, 0, width, height);
  }
  readTexture(texture, width, height, dataSize, dataType, channels) {
    const gl = this.gl;
    if (!channels) {
      channels = 1;
    }
    if (!this.frameBufferBound) {
      this.attachFramebuffer(texture, width, height);
    }
    const encoder = this.getEncoder(dataType, channels);
    const buffer = encoder.allocate(width * height);
    // bind texture to framebuffer
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0); // 0, we aren't using MIPMAPs
    // TODO: Check if framebuffer is ready
    gl.readPixels(0, 0, width, height, gl.RGBA, encoder.textureType, buffer);
    this.checkError();
    // unbind FB
    return encoder.decode(buffer, dataSize);
  }
  isFramebufferReady() {
    // TODO: Implement logic to check if the framebuffer is ready
    return true;
  }
  getActiveTexture() {
    const gl = this.gl;
    const n = gl.getParameter(this.gl.ACTIVE_TEXTURE);
    return `TEXTURE${(n - gl.TEXTURE0)}`;
  }
  getTextureBinding() {
    return this.gl.getParameter(this.gl.TEXTURE_BINDING_2D);
  }
  getFramebufferBinding() {
    return this.gl.getParameter(this.gl.FRAMEBUFFER_BINDING);
  }
  setVertexAttributes(positionHandle, textureCoordHandle) {
    const gl = this.gl;
    gl.vertexAttribPointer(positionHandle, 3, gl.FLOAT, false, 20, 0);
    gl.enableVertexAttribArray(positionHandle);
    if (textureCoordHandle !== -1) {
      gl.vertexAttribPointer(textureCoordHandle, 2, gl.FLOAT, false, 20, 12);
      gl.enableVertexAttribArray(textureCoordHandle);
    }
    this.checkError();
  }
  createProgram(vertexShader, fragShader) {
    const gl = this.gl;
    const program = gl.createProgram();
    // the program consists of our shaders
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragShader);
    gl.linkProgram(program);
    return program;
  }
  compileShader(shaderSource, shaderType) {
    const gl = this.gl;
    const shader = gl.createShader(shaderType);
    if (!shader) {
      throw new Error(`createShader() returned null with type ${shaderType}`);
    }
    gl.shaderSource(shader, shaderSource);
    gl.compileShader(shader);
    if (gl.getShaderParameter(shader, gl.COMPILE_STATUS) === false) {
      throw new Error(`Failed to compile shader: ${gl.getShaderInfoLog(shader)}`);
    }
    return shader;
  }
  deleteShader(shader) {
    this.gl.deleteShader(shader);
  }
  bindTextureToUniform(texture, position, uniformHandle) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + position);
    this.checkError();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    this.checkError();
    gl.uniform1i(uniformHandle, position);
    this.checkError();
  }
  draw() {
    this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
    this.checkError();
  }
  checkError() {
    const gl = this.gl;
    const error = gl.getError();
    let label = '';
    switch (error) {
      case (gl.NO_ERROR):
        return;
      case (gl.INVALID_ENUM):
        label = 'INVALID_ENUM';
        break;
      case (gl.INVALID_VALUE):
        label = 'INVALID_VALUE';
        break;
      case (gl.INVALID_OPERATION):
        label = 'INVALID_OPERATION';
        break;
      case (gl.INVALID_FRAMEBUFFER_OPERATION):
        label = 'INVALID_FRAMEBUFFER_OPERATION';
        break;
      case (gl.OUT_OF_MEMORY):
        label = 'OUT_OF_MEMORY';
        break;
      case (gl.CONTEXT_LOST_WEBGL):
        label = 'CONTEXT_LOST_WEBGL';
        break;
      default:
        label = 'Unknown WebGL Error: ' + error.toString(16);
    }
    throw new Error(label);
  }
  deleteTexture(texture) {
    this.gl.deleteTexture(texture);
  }
  deleteProgram(program) {
    this.gl.deleteProgram(program);
  }
  getEncoder(dataType, channels, usage = 0 /* Default */) {
    if (this.version === 2) {
      return new RedFloat32DataEncoder(this.gl, channels);
    }
    switch (dataType) {
      case 'float':
        if (usage === 1 /* UploadOnly */ || this.isRenderFloat32Supported) {
          return new RGBAFloatDataEncoder(this.gl, channels);
        }
        else {
          return new RGBAFloatDataEncoder(this.gl, channels, this.textureHalfFloatExtension.HALF_FLOAT_OES);
        }
      case 'int':
        throw new Error('not implemented');
      case 'byte':
        return new Uint8DataEncoder(this.gl, channels);
      default:
        throw new Error(`Invalid dataType: ${dataType}`);
    }
  }
  clearActiveTextures() {
    const gl = this.gl;
    for (let unit = 0; unit < this.maxTextureImageUnits; ++unit) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, null);
    }
  }
  dispose() {
    if (this.disposed) {
      return;
    }
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(this.framebuffer);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.deleteBuffer(this.vertexbuffer);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
    gl.finish();
    this.disposed = true;
  }
  createDefaultGeometry() {
    // Sets of x,y,z(=0),s,t coordinates.
    return new Float32Array([
      -1.0, 1.0, 0.0, 0.0, 1.0,
      -1.0, -1.0, 0.0, 0.0, 0.0,
      1.0, 1.0, 0.0, 1.0, 1.0,
      1.0, -1.0, 0.0, 1.0, 0.0
    ]); // lower right
  }
  createVertexbuffer() {
    const gl = this.gl;
    const buffer = gl.createBuffer();
    if (!buffer) {
      throw new Error('createBuffer() returned null');
    }
    const geometry = this.createDefaultGeometry();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, geometry, gl.STATIC_DRAW);
    this.checkError();
    return buffer;
  }
  createFramebuffer() {
    const fb = this.gl.createFramebuffer();
    if (!fb) {
      throw new Error('createFramebuffer returned null');
    }
    return fb;
  }
  queryVitalParameters() {
    const gl = this.gl;
    this.isFloatTextureAttachableToFrameBuffer = this.checkFloatTextureAttachableToFrameBuffer();
    this.isRenderFloat32Supported = this.checkRenderFloat32();
    this.isFloat32DownloadSupported = this.checkFloat32Download();
    if (this.version === 1 && !this.textureHalfFloatExtension && !this.isRenderFloat32Supported) {
      throw new Error(`both float32 and float16 TextureType are not supported`);
    }
    this.isBlendSupported = !this.isRenderFloat32Supported || this.checkFloat32Blend();
    this.maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    this.maxTextureImageUnits = gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS);
  }
  getExtensions() {
    if (this.version === 2) {
      this.colorBufferFloatExtension = this.gl.getExtension('EXT_color_buffer_float');
    }
    else {
      this.textureFloatExtension = this.gl.getExtension('OES_texture_float');
      this.textureHalfFloatExtension = this.gl.getExtension('OES_texture_half_float');
    }
  }
  checkFloatTextureAttachableToFrameBuffer() {
    // test whether Float32 texture is supported:
    // STEP.1 create a float texture
    const gl = this.gl;
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    const internalFormat = this.version === 2 ? gl.RGBA32F : gl.RGBA;
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, 1, 1, 0, gl.RGBA, gl.FLOAT, null);
    // STEP.2 bind a frame buffer
    const frameBuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);
    // STEP.3 attach texture to framebuffer
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    // STEP.4 test whether framebuffer is complete
    const isComplete = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteTexture(texture);
    gl.deleteFramebuffer(frameBuffer);
    return isComplete;
  }
  checkRenderFloat32() {
    if (this.version === 2) {
      if (!this.colorBufferFloatExtension) {
        return false;
      }
    }
    else {
      if (!this.textureFloatExtension) {
        return false;
      }
    }
    return this.isFloatTextureAttachableToFrameBuffer;
  }
  checkFloat32Download() {
    if (this.version === 2) {
      if (!this.colorBufferFloatExtension) {
        return false;
      }
    }
    else {
      if (!this.textureFloatExtension) {
        return false;
      }
      if (!this.gl.getExtension('WEBGL_color_buffer_float')) {
        return false;
      }
    }
    return this.isFloatTextureAttachableToFrameBuffer;
  }
  /**
   * Check whether GL_BLEND is supported
   */
  checkFloat32Blend() {
    // it looks like currently (2019-05-08) there is no easy way to detect whether BLEND is supported
    // https://github.com/microsoft/onnxjs/issues/145
    const gl = this.gl;
    let texture;
    let frameBuffer;
    let vertexShader;
    let fragmentShader;
    let program;
    try {
      texture = gl.createTexture();
      frameBuffer = gl.createFramebuffer();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      const internalFormat = this.version === 2 ? gl.RGBA32F : gl.RGBA;
      gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, 1, 1, 0, gl.RGBA, gl.FLOAT, null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
      gl.enable(gl.BLEND);
      vertexShader = gl.createShader(gl.VERTEX_SHADER);
      if (!vertexShader) {
        return false;
      }
      gl.shaderSource(vertexShader, 'void main(){}');
      gl.compileShader(vertexShader);
      fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
      if (!fragmentShader) {
        return false;
      }
      gl.shaderSource(fragmentShader, `precision highp float;void main(){gl_FragColor=vec4(0.5);}`);
      gl.compileShader(fragmentShader);
      program = gl.createProgram();
      if (!program) {
        return false;
      }
      gl.attachShader(program, vertexShader);
      gl.attachShader(program, fragmentShader);
      gl.linkProgram(program);
      gl.useProgram(program);
      gl.drawArrays(gl.POINTS, 0, 1);
      return gl.getError() === gl.NO_ERROR;
    }
    finally {
      gl.disable(gl.BLEND);
      if (program) {
        gl.deleteProgram(program);
      }
      if (vertexShader) {
        gl.deleteShader(vertexShader);
      }
      if (fragmentShader) {
        gl.deleteShader(fragmentShader);
      }
      if (frameBuffer) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.deleteFramebuffer(frameBuffer);
      }
      if (texture) {
        gl.bindTexture(gl.TEXTURE_2D, null);
        gl.deleteTexture(texture);
      }
    }
  }
}