#include "common.h"
#include "forward.h"

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
      PARAM_INT32_PTR(data, dataIndex[11]), PARAM_INT32(data, dataIndex[12]),
      PARAM_FLOAT_PTR(data, dataIndex[13]), PARAM_FLOAT_PTR(data, dataIndex[14]),
      PARAM_FLOAT_PTR(data, dataIndex[15]));
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
      PARAM_INT32_PTR(data, dataIndex[11]), PARAM_INT32(data, dataIndex[12]),
      PARAM_FLOAT_PTR(data, dataIndex[13]), PARAM_FLOAT_PTR(data, dataIndex[14]),
      PARAM_FLOAT_PTR(data, dataIndex[15]));
}
void matmul_f32(void *data)
{
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  uint32_t const argc = dataIndex[0];
  connected_imp(
      PARAM_FLOAT_PTR(data, dataIndex[1]), PARAM_INT32_PTR(data, dataIndex[2]),
      PARAM_FLOAT_PTR(data, dataIndex[3]), PARAM_INT32_PTR(data, dataIndex[4]),
      PARAM_FLOAT_PTR(data, dataIndex[5]), PARAM_INT32_PTR(data, dataIndex[6]),
      PARAM_FLOAT_PTR(data, dataIndex[7]), PARAM_INT32(data, dataIndex[8]),
      PARAM_FLOAT_PTR(data, dataIndex[9]),PARAM_FLOAT_PTR(data, dataIndex[11]),
      PARAM_FLOAT_PTR(data, dataIndex[11]));
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