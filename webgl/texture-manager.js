"use strict";
/**
 * TextureManager is the mainly responsible for caching Textures
 * Textures are cached in 2 levels:
 *   1. the texures which are associated with a dataId (from Tensor)
 *    Caching these is crucial to performance. These are In-use Textures
 *   2. textures which are not in use by any current ProgramInfo/Tensor
 *     These are called Free Textures
 * TextureManager is also used to help creating textures. For this it
 * uses WebGLContext and TextureLayoutStrategy
 */
class TextureManager {
  constructor(glContext, reuseTextures) {
    this.glContext = glContext;
    this.reuseTextures = reuseTextures;
    if (reuseTextures) {
      this.inUseTextures = new Map();
      this.idleTextures = new Map();
      this.textureLookup = new Map();
    }
  }
  toTextureData(dataType, data) {
    if (!data) {
        return undefined;
    }
    return (data instanceof Float32Array) ? data : new Float32Array(data);
  }
  createTextureFromLayout(dataType, layout, data, usage) {
    const textureDataType = 'float';
    const encoder = this.glContext.getEncoder(textureDataType, layout.channels || 1, usage);
    let key;
    let inUseTextures;
    if (this.reuseTextures) {
      key = `${layout.width}x${layout.height}_${encoder.format}_${encoder.internalFormat}_${encoder.textureType}`;
      inUseTextures = this.inUseTextures.get(key);
      if (!inUseTextures) {
        inUseTextures = [];
        this.inUseTextures.set(key, inUseTextures);
      }
      const idleTextures = this.idleTextures.get(key);
      if (idleTextures && idleTextures.length > 0) {
        const texture = idleTextures.pop();
        inUseTextures.push(texture);
        if (usage === 1 /* UploadOnly */) {
          this.glContext.updateTexture(texture, layout.width, layout.height, encoder, this.toTextureData(dataType, data));
        }
        return texture;
      }
    }
    console.log('TextureManager', `Creating new texture of size ${layout.width}x${layout.height}`);
    const texture = this.glContext.allocateTexture(layout.width, layout.height, encoder, this.toTextureData(dataType, data));
    if (this.reuseTextures) {
      inUseTextures.push(texture);
      this.textureLookup.set(texture, key);
    }
    return texture;
  }
  readTexture(td, dataType, channels) {
    if (!channels) channels = 1;
  //  return this.profiler.event('backend', 'TextureManager.readTexture', () => {
      const dataSize = td.shape.reduce((a, b) => a * b) * channels;
      return this.glContext.readTexture(td.texture, td.width, td.height, dataSize, dataType, channels);
  //  });
  }
  readUint8TextureAsFloat(td) {
   // return this.profiler.event('backend', 'TextureManager.readUint8TextureAsFloat', () => {
      const dataSize = td.shape.reduce((a, b) => a * b);
      const data = this.glContext.readTexture(td.texture, td.width, td.height, dataSize * 4, 'byte', 4);
      return new Float32Array(data.buffer, data.byteOffset, dataSize);
   // });
  }
  releaseTexture(textureData, deleteTexture) {
    let key;
    if (this.config.reuseTextures) {
      key = this.textureLookup.get(textureData.texture);
      if (key) {
        if (deleteTexture) {
          this.textureLookup.delete(key);
        }
        const inUseTextures = this.inUseTextures.get(key);
        if (inUseTextures) {
          const index = inUseTextures.indexOf(textureData.texture);
          if (index !== -1) {
            inUseTextures.splice(index, 1);
            let idleTextures = this.idleTextures.get(key);
            if (!idleTextures) {
              idleTextures = [];
              this.idleTextures.set(key, idleTextures);
            }
            idleTextures.push(textureData.texture);
          }
        }
      }
    }
    if (!key || deleteTexture) {
      console.log('TextureManager', `Deleting texture of size ${textureData.width}x${textureData.height}`);
      this.glContext.deleteTexture(textureData.texture);
    }
  }
  clearActiveTextures() {
    this.glContext.clearActiveTextures();
  }
}