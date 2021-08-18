#include "common.h"
//#include "conv.h"
#include "forward.h"
/*
W:\nets\emsdk\upstream\emscripten\emcc.bat -g forward.cpp -o wforward.js -s WASM=1 -s NO_EXIT_RUNTIME=0 -s ALLOW_MEMORY_GROWTH=1 -s SAFE_HEAP=0 -s SAFE_HEAP_LOG=0 -s STACK_OVERFLOW_CHECK=0  -s EXPORTED_FUNCTIONS="['_malloc','_free','_conv_f32','_convT_f32','_pool_f32']" -std=c++11 -s EXPORT_ALL=0 -O3 -msimd128
/media/user/4C35EEFA105334CC/nets/emsdkl/emsdk/upstream/emscripten/emcc -g forward.cpp -o wforward.js -s WASM=1 -s NO_EXIT_RUNTIME=0 -s ALLOW_MEMORY_GROWTH=1 -s SAFE_HEAP=0 -s SAFE_HEAP_LOG=0 -s STACK_OVERFLOW_CHECK=0  -s EXPORTED_FUNCTIONS="['_malloc','_free','_conv_f32','_convT_f32','_pool_f32']" -std=c++11 -s EXPORT_ALL=0 -O3 -msimd128
*/
// Wasm interop method
void conv_f32(void *data)
{
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  uint32_t const argc = dataIndex[0];
  conv2D_f32_imp(
      PARAM_FLOAT_PTR(data, dataIndex[1]), PARAM_INT32_PTR(data, dataIndex[2]),
      PARAM_FLOAT_PTR(data, dataIndex[3]), PARAM_INT32_PTR(data, dataIndex[4]),
      PARAM_FLOAT_PTR(data, dataIndex[5]), PARAM_INT32_PTR(data, dataIndex[6]),
      PARAM_FLOAT_PTR(data, dataIndex[7]), PARAM_INT32_PTR(data, dataIndex[8]),
      PARAM_INT32(data, dataIndex[9]), PARAM_INT32_PTR(data, dataIndex[10]),
      PARAM_INT32_PTR(data, dataIndex[11]), PARAM_INT32(data, dataIndex[12]));
}
void convT_f32(void *data)
{
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  uint32_t const argc = dataIndex[0];
  convTranspose2D_f32_imp(
      PARAM_FLOAT_PTR(data, dataIndex[1]), PARAM_INT32_PTR(data, dataIndex[2]),
      PARAM_FLOAT_PTR(data, dataIndex[3]), PARAM_INT32_PTR(data, dataIndex[4]),
      PARAM_FLOAT_PTR(data, dataIndex[5]), PARAM_INT32_PTR(data, dataIndex[6]),
      PARAM_FLOAT_PTR(data, dataIndex[7]), PARAM_INT32_PTR(data, dataIndex[8]),
      PARAM_INT32(data, dataIndex[9]), PARAM_INT32_PTR(data, dataIndex[10]),
      PARAM_INT32_PTR(data, dataIndex[11]), PARAM_INT32(data, dataIndex[12]));
}
void pool_f32(void *data)
{
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  uint32_t const argc = dataIndex[0];
  pool2D_f32_imp(
      PARAM_FLOAT_PTR(data, dataIndex[1]), PARAM_INT32_PTR(data, dataIndex[2]),
      PARAM_FLOAT_PTR(data, dataIndex[3]), PARAM_INT32_PTR(data, dataIndex[4]),
      PARAM_INT32(data, dataIndex[5]), PARAM_INT32_PTR(data, dataIndex[6]),
      PARAM_INT32_PTR(data, dataIndex[7]), PARAM_BOOL(data, dataIndex[8]));
}