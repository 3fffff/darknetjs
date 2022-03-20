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
  constructor(glContext) {
    this.glContext = glContext;
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
  readTexture(td, dataType, channels) {
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
}