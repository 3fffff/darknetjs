"use strict";
/**
 * ProgramManager is the main class behind running computations
 * It builds ProgramInfo's into Artifacts
 * It compiles given ProgramInfo's into WebGL Programs (cached as Artifacts)
 * Uses the artifact to run the computation by calling Draw on
 * the WebGL drawing buffer
 * ProgramManager automatically maps (binds) input variables to their
 * corresponding Location's in the binary program
 */
class ProgramManager {
  constructor(glContext) {
    this.debug = false
    this.glContext = glContext;
    this.attributesBound = false;
  }
  getTextureData(tensorId) {
    return this.textureDataCache.get(tensorId);
  }
  setTextureData(tensorId, textureData) {
    console.log('Storing Texture data in cache');
    this.textureDataCache.set(tensorId, textureData);
  }
  run(buildArtifact, runData) {
    // this.profiler.event('ProgramManager', 'ProgramManager.run', () => {
    const gl = this.glContext.gl;
    const program = buildArtifact.program;
    gl.useProgram(program);
    try {
      this.bindOutput(runData.outputTextureData);
      if (!this.attributesBound) this.bindAttributes(buildArtifact.attribLocations);
      this.bindUniforms(buildArtifact.uniformLocations, runData.uniformData, runData.inputTextureDatas);
    }
    catch (err) {
      console.log('ProgramManager', buildArtifact.programInfo.shaderSource);
      throw err;
    }
    //this.profiler.event('ProgramManager', 'GlContext.draw()', () => {
    this.doDraw(buildArtifact, runData);
    gl.flush();
    //});
    // });
  }
  dispose() {
    if (this.vertexShader) this.glContext.deleteShader(this.vertexShader);
    this.repo.forEach(a => this.glContext.deleteProgram(a.program));
  }
  build(programInfo) {
    //return this.profiler.event('ProgramManager', 'ProgramManager.build', () => {
    const preprocessor = new GlslPreprocessor(this.glContext, programInfo);
    const fragScript = preprocessor.preprocess();
    const program = this.compile(fragScript);
    const artifact = {
      programInfo,
      program,
      uniformLocations: this.getUniformLocations(program, preprocessor.context.programInfo.samplers, preprocessor.context.programInfo.variables),
      attribLocations: this.getAttribLocations(program)
    };
    return artifact;
    //  });
  }
  doDraw(artifact, runData) {
    if (runData.draw) {
      if (this.debug)
        console.log('ProgramManager', 'Custom draw function');
      runData.draw(this.glContext, artifact);
    }
    else {
      this.glContext.draw();
    }
  }
  compile(fragShaderScript) {
    if (!this.vertexShader) {
      if (this.debug) console.log('ProgramManager', 'Compiling and caching Vertex shader for the first time');
      const vertexShaderScript = getVertexShaderSource(this.glContext.version);
      this.vertexShader = this.glContext.compileShader(vertexShaderScript, this.glContext.gl.VERTEX_SHADER);
    }
    if (this.debug) console.log('ProgramManager', `FragShader:${fragShaderScript}`);
    const fragShader = this.glContext.compileShader(fragShaderScript, this.glContext.gl.FRAGMENT_SHADER);
    const program = this.glContext.createProgram(this.vertexShader, fragShader);
    this.glContext.deleteShader(fragShader);
    return program;
  }
  bindOutput(td) {
    if (this.debug) console.log('ProgramManager', `Binding output texture to Framebuffer: w/h=${td.width}/${td.height}, shape=${td.channels}`);
    this.glContext.attachFramebuffer(td.texture, td.width, td.height);
  }
  bindAttributes(attribLocations) {
    const positionHandle = attribLocations.position;
    const textureCoordHandle = attribLocations.textureCoord;
    this.glContext.setVertexAttributes(positionHandle, textureCoordHandle);
    this.attributesBound = true;
  }
  bindUniforms(uniformLocations, uniformData, textures) {
    const gl = this.glContext.gl;
    let texturePosition = 0;
    for (const { name, type, location, arrayLength } of uniformLocations) {
      switch (type) {
        case 'sampler2D':
          this.bindTexture(textures[texturePosition], location, texturePosition);
          texturePosition++;
          break;
        case 'float':
          if (arrayLength) {
            gl.uniform1fv(location, uniformData[name]);
          }
          else {
            gl.uniform1f(location, uniformData[name]);
          }
          break;
        case 'int':
          if (arrayLength) {
            gl.uniform1iv(location, uniformData[name]);
          }
          else {
            gl.uniform1i(location, uniformData[name]);
          }
          break;
        default:
          throw new Error(`Uniform not implemented: ${type}`);
      }
    }
  }
  bindTexture(td, uniformHandle, position) {
    this.glContext.bindTextureToUniform(td.texture, position, uniformHandle);
  }
  getAttribLocations(program) {
    return {
      position: this.getAttribLocation(program, 'position'),
      textureCoord: this.getAttribLocation(program, 'textureCoord')
    };
  }
  getUniformLocations(program, samplers, variables) {
    const uniformLocations = [];
    if (samplers) {
      for (const sampler of samplers) {
        uniformLocations.push({ name: sampler, type: 'sampler2D', location: this.getUniformLocation(program, sampler) });
      }
    }
    if (variables) {
      for (const variable of variables) {
        uniformLocations.push(Object.assign(Object.assign({}, variable), { location: this.getUniformLocation(program, variable.name) }));
      }
    }
    return uniformLocations;
  }
  getUniformLocation(program, name) {
    const gl = this.glContext.gl;
    const reference = gl.getUniformLocation(program, name);
    if (reference === null) {
      throw new Error(`Uniform ${name} not found.`);
    }
    return reference;
  }
  getAttribLocation(program, name) {
    const gl = this.glContext.gl;
    const attributeLocation = gl.getAttribLocation(program, name);
    return attributeLocation;
  }
}