"use strict";
class WebGLSoftmax {
  createSoftMaxProgramInfo(inferenceHandler, input, N, D, maxElementPerLogicalRow, normalizationPerLogicalRow) {
    const inputShape = input.dims.slice();
    const inputLayout = inferenceHandler.createTextureLayoutFromShape(inputShape);
    const outputShape = inputShape;
    const rank = outputShape.length;
    const textureWidth = inputLayout.width;
    const textureHeight = inputLayout.height;
    if (N < 1 || D < 1) {
      throw new Error(`Logical row count N and feature count D must be greater than or equal to 1`);
    }
    if (maxElementPerLogicalRow.shape.length !== 1 || normalizationPerLogicalRow.shape.length !== 1) {
      throw new Error(`Dimensionality of the intermediate results should be 1`);
    }
    if (maxElementPerLogicalRow.shape[0] !== N || normalizationPerLogicalRow.shape[0] !== N) {
      throw new Error(`Shape of the intermediate results should be equal to logical row count`);
    }
    const shaderSource = `
    float process(int[${rank}] indices) {

      // get offset of current logical tensor index from the 2-D texture coordinates (TexCoords)
      int offset = coordsToOffset(TexCoords, ${textureWidth}, ${textureHeight});

      //determine the logical row for this index
      int logical_row_index[1];
      logical_row_index[0] = offset / ${D};

      float norm_factor = _Norm(logical_row_index);

      // avoid possible division by 0
      // if norm_facor is 0, all elements are zero
      // if so, return 0
      if(norm_factor == 0.0)
        return 0.0;

      return exp(_A(indices) - _Max(logical_row_index)) / norm_factor;
    }`;
    return {
      inputLayouts: [inputLayout, maxElementPerLogicalRow, normalizationPerLogicalRow],
      outputLayout: inferenceHandler.createTextureLayoutFromShape(outputShape),
      samplers: ['A', 'Max', 'Norm'],
      shaderSource,
    };
  }
  /**
   * Create a texture that contains the normalization factor for each of the 'N' rows
   */
  createComputScaleProgramInfo(inferenceHandler, x, N, D, maxElementPerLogicalRow, outputShape) {
    const xlayout = inferenceHandler.createTextureLayoutFromShape(x.dims.slice());
    const rank = outputShape.length;
    const textureWidth = xlayout.width;
    const textureHeight = xlayout.height;
    if (N < 1 || D < 1) throw new Error(`Logical row count N and feature count D must be greater than or equal to 1`);
    if (outputShape.length !== 1) throw new Error(`Dimensionality of the output should be 1`);
    if (outputShape[0] !== N) throw new Error(`Shape of the output should be equal to logical row count`);
    if (maxElementPerLogicalRow.shape.length !== 1) throw new Error(`Dimensionality of the intermediate results should be 1`);
    if (maxElementPerLogicalRow.shape[0] !== N) throw new Error(`Shape of the intermediate results should be equal to logical row count`);
    const glsl = getGlsl(inferenceHandler.session.backend.glContext.version);
    const shaderSource = `
    float process(int[${rank}] indices) {

      int logical_row_start_offset = indices[0] * ${D};

      float norm_factor = 0.0;
      float max = _Max(indices);
      for(int i=0; i<${D}; ++i)
      {
        norm_factor += exp(getColorAsFloat(${glsl.texture2D}(A, offsetToCoords(logical_row_start_offset + i, ${textureWidth}, ${textureHeight}))) - max);
      }

      return norm_factor;
    }`;
    return {
      inputLayouts: [xlayout, maxElementPerLogicalRow],
      outputLayout: inferenceHandler.createTextureLayoutFromShape(outputShape),
      samplers: ['A', 'Max'],
      shaderSource,
    };
  }
  /**
   * Create a texture that contains the maximum value of each of the 'N' rows
   */
  createComputeMaxProgramInfo(inferenceHandler, x, N, D, outputShape) {
    const xlayout = inferenceHandler.createTextureLayoutFromShape(x.dims.slice());
    const rank = outputShape.length;
    const textureWidth = xlayout.width;
    const textureHeight = xlayout.height;
    if (N < 1 || D < 1) {
      throw new Error(`Logical row count N and feature count D must be greater than or equal to 1`);
    }
    if (outputShape.length !== 1) {
      throw new Error(`Dimensionality of the output should be 1`);
    }
    if (outputShape[0] !== N) {
      throw new Error(`Shape of the output should be equal to logical row count`);
    }
    const glsl = getGlsl(inferenceHandler.session.backend.glContext.version);
    const shaderSource = `
        float process(int[${rank}] indices) {

          int logical_row_start_offset = indices[0] * ${D};

          float max = getColorAsFloat(${glsl.texture2D}(A, offsetToCoords(logical_row_start_offset, ${textureWidth}, ${textureHeight} )));
          for(int i=1; i<${D}; ++i)
          {
            float current = getColorAsFloat(${glsl.texture2D}(A, offsetToCoords(logical_row_start_offset + i, ${textureWidth}, ${textureHeight})));
            if(current > max)
              max = current;
          }

          return max;
        }`;
    return {
      inputLayouts: [xlayout],
      outputLayout: inferenceHandler.createTextureLayoutFromShape(outputShape),
      samplers: ['A'],
      shaderSource,
    };
  }
  createProgramInfos(inferenceHandler, inputs) {
    const axis = WebGLSoftmax.normalizeAxis(this.axis, input.shape.length);
    const N = WebGLSoftmax.sizeToDimension(input.shape, axis);
    const D = WebGLSoftmax.sizeFromDimension(input.shape, axis);
    const computeMaxProgramInfo = this.createComputeMaxProgramInfo(inferenceHandler, inputs[0], N, D, [N]);
    const computeScaleProgramInfo = this.createComputScaleProgramInfo(inferenceHandler, inputs[0], N, D, computeMaxProgramInfo.outputLayout, [N]);
    const softMaxProgramInfo = this.createSoftMaxProgramInfo(inferenceHandler, inputs[0], N, D, computeMaxProgramInfo.outputLayout, computeScaleProgramInfo.outputLayout);
    const programInfos = [computeMaxProgramInfo, computeScaleProgramInfo, softMaxProgramInfo];
    return programInfos;
  }
  createRunDatas(inferenceHandler, programInfos, inputs) {
    const dataType = inputs[0].type;
    const inputTD = inferenceHandler.getOrCreateTextureData(inputs[0], programInfos[0].inputLayouts[0]);
    const runDatas = [];
    runDatas.push({
      inputTextureDatas: [inputTD],
      outputTextureData: inferenceHandler.createTextureDataFromLayout(programInfos[0].outputLayout, dataType),
      uniformData: {}
    });
    for (let i = 1; i < programInfos.length; ++i) {
      runDatas.push({
        inputTextureDatas: [...runDatas[i - 1].inputTextureDatas, runDatas[i - 1].outputTextureData],
        outputTextureData: inferenceHandler.createTextureDataFromLayout(programInfos[i].outputLayout, dataType),
        uniformData: {}
      });
    }
    return runDatas;
  }
  static normalizeAxis(axis, tensorRank) {
    if (axis < -tensorRank && axis >= tensorRank) {
      throw new Error('unsupported axis for this operation.');
    }
    return axis < 0 ? axis + tensorRank : axis;
  }
  // `axis` inclusive
  static sizeFromDimension(dims, axis) {
    if (axis < 0 || axis > dims.length) {
      throw new Error(`invalid dimension of ${axis} for sizeFromDimension as Tensor has ${dims.length} dimensions.`);
    }
    return WebGLSoftmax.getSizeFromDimensionRange(dims, axis, dims.length);
  }

  static getSizeFromDimensionRange(dims, start, end) {
    let size = 1;
    for (let i = start; i < end; i++) {
      // safety check as this method is called by multiple other methods requiring size.
      // size cannot be 0 or negative.
      if (dims[i] <= 0) {
        throw new Error(
          // tslint:disable-next-line:max-line-length
          `cannot get valid size from specified dimension range. Most likely the range contains 0 or negative values in them.`);
      }
      size *= dims[i];
    }
    return size;
  }
}