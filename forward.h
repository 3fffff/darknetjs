#pragma once
#include <vector>
#include <algorithm>
#include <limits>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <mm_malloc.h>
#include <wasm_simd128.h>

extern "C"
{
  void conv_f32(void *);
  void convT_f32(void *);
  void pool_f32(void *);

  void conv2D_f32_imp(float *, int32_t *, float *, int32_t *, float *, int32_t *,
                      float *, int32_t *, int32_t, int32_t *, int32_t *, int32_t, bool);
  void convTranspose2D_f32_imp(float *, int32_t *, float *, int32_t *, float *, int32_t *,
                               float *, int32_t *, int32_t, int32_t *, int32_t *, int32_t, bool);
  void im2col_f32(const float *, const int32_t, const int32_t, const int32_t,
                  const int32_t, const int32_t, const int32_t, const int32_t,
                  const int32_t, const int32_t, const int32_t, const int32_t,
                  const int32_t, const int32_t, float *);
  void matmul(float *, float *, float *,
              const int32_t, const int32_t, const int32_t);
  void pool2D_f32_imp(float *, int *, float *, int *, int, int *, int *, bool);
  float *activate(float *, const int, const int);
  void addBias(float *, const float *, const int32_t, const int32_t);
  void micro4x8(size_t, const float *, size_t, const float *, size_t, float *, size_t);
  void reorder_b_8(int, const float *, int, float *);
  //void matmul4x8(float *, float *, float *, int, int, int);
  /* void conv3x3S1(float *const &, const int, const int, const int,
                 float *&, const int &, const int &, const int &, float *const &);
  void conv3x3S2(float *const &, const int &, const int &, const int &,
                 float *&, const int &, const int &, const int &, float *const &);
  void conv1x1s1(float *const &, const int &, const int &, const int &,
                 float *&, const int &, const int &, const int &, float *const &);
  void conv1x1s2(float *const &, const int &, const int &, const int &,
                 float *&, const int &, const int &, const int &, float *const &);
  void convdw3x3s1(float *const &, const int &, const int &, const int &,
                   float *&, const int &, const int &, const int &, float *const &, float);
  void convdw3x3s2(float *const &, const int &, const int &, const int &,
                   float *&, const int &, const int &, const int &, float *const &, float);*/
}
struct buf_t
{
  float *p;
  int n;
  buf_t(int size) : n(size), p((float *)_mm_malloc(size * 4, 64)) {}
  ~buf_t() { _mm_free(p); }
};
// Core operator implementation
void conv2D_f32_imp(float *X, int *X_shape, float *W, int *W_shape, float *Y,
                    int *Y_shape, float *bias, int *dilations, int groups,
                    int *pads, int *strides, int active)
{
  const int input_num = X_shape[0];
  const int input_channels = X_shape[1];
  const int input_height = X_shape[2];
  const int input_width = X_shape[3];
  const int input_size =
      input_num * input_channels * input_height * input_width;

  const int filter_num = W_shape[0];
  const int filter_channels = W_shape[1];
  const int filter_height = W_shape[2];
  const int filter_width = W_shape[3];
  const int filter_size =
      filter_num * filter_channels * filter_height * filter_width;

  const int output_num = Y_shape[0];
  const int output_channels = Y_shape[1];
  const int output_height = Y_shape[2];
  const int output_width = Y_shape[3];
  const int output_size =
      output_num * output_channels * output_height * output_width;

  const int output_image_size = output_height * output_width;
  const int X_offset = input_channels / groups * input_height * input_width;
  const int Y_offset = output_size / output_num / groups;
  const int W_offset = filter_size / groups;
  const int kernel_dim = input_channels / groups * filter_height * filter_width;
  const int col_buffer_size = kernel_dim * output_image_size;
  const int out_size = output_height * output_width;
  float *col_buffer_data = new float[col_buffer_size];

  for (int n = 0; n < input_num; ++n)
  {
    for (int group = 0; group < groups; ++group)
    {
      im2col_f32(X + X_offset * group, input_channels / groups, input_height,
                 input_width, filter_height, filter_width, dilations[0],
                 dilations[1], pads[0], pads[1], pads[2], pads[3], strides[0],
                 strides[1], col_buffer_data);
      matmul(W + W_offset * group, col_buffer_data, Y + Y_offset * group, filter_num / groups, output_height * output_width, kernel_dim);
    }
  }
  if (bias != nullptr)
    addBias(Y, bias, filter_num, out_size);
  Y = activate(Y, output_size, active);
  delete[] col_buffer_data;
}

void convTranspose2D_f32_imp(float *X, int *X_shape, float *W, int *W_shape,
                             float *Y, int *Y_shape, float *bias, int *dilations,
                             int groups, int *pads, int *strides, int active)
{

  const int input_num = X_shape[0];
  const int input_channels = X_shape[1];
  const int input_height = X_shape[2];
  const int input_width = X_shape[3];
  const int input_size =
      input_num * input_channels * input_height * input_width;

  const int filter_num = W_shape[0];
  const int filter_channels = W_shape[1];
  const int filter_height = W_shape[2];
  const int filter_width = W_shape[3];
  const int filter_size =
      filter_num * filter_channels * filter_height * filter_width;

  const int output_num = Y_shape[0];
  const int output_channels = Y_shape[1];
  const int output_height = Y_shape[2];
  const int output_width = Y_shape[3];
  const int output_size =
      output_num * output_channels * output_height * output_width;

  const int input_image_size = input_height * input_width;
  const int output_image_size = output_height * output_width;
  const int kernel_size = filter_height * filter_width;
  const int X_offset = input_channels / groups * input_image_size;
  const int Y_offset = output_size / output_num / groups;
  const int W_offset = filter_size / groups;
  const int kernel_dim = input_channels / groups * kernel_size;
  const int out_size = output_height * output_width;
  const int col_buffer_size = kernel_dim * output_image_size;

  float *col_buffer_data = new float[kernel_dim * input_image_size];

  for (auto n = 0; n < input_num; ++n)
  {
    for (int group = 0; group < groups; ++group)
    {
      // Weight term
      matmul(W + W_offset * group, X + X_offset * group, col_buffer_data, filter_num / groups, output_image_size, kernel_dim);
      // Col2im
      im2col_f32(col_buffer_data, output_channels / groups,
                 output_height, output_width, filter_height, filter_width,
                 dilations[0], dilations[1], pads[0], pads[1], pads[2], pads[3],
                 strides[0], strides[1], Y + Y_offset * group);
    }
  }
  if (bias != nullptr)
    addBias(Y, bias, filter_num, out_size);
  Y = activate(Y, output_size, active);
  delete[] col_buffer_data;
}

// Some helpers specific to conv operator
void im2col_f32(const float *data_im, const int channels, const int height,
                const int width, const int kernel_h, const int kernel_w,
                const int dilation_h, const int dilation_w, const int pad_t,
                const int pad_l, const int pad_b, const int pad_r,
                const int stride_h, const int stride_w, float *data_col)
{
  const int output_h =
      (height + pad_b + pad_t - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w =
      (width + pad_l + pad_r - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  // Baseline
  const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int dkernel_w = dilation_w * (kernel_w - 1) + 1;

  int height_col = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;

  int channels_col = channels * kernel_h * kernel_w;
  for (int c = 0; c < channels_col; ++c)
  {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / kernel_h / kernel_w;
    for (int h = 0; h < height_col; ++h)
    {
      for (int w = 0; w < width_col; ++w)
      {
        int h_pad = h * stride_h - pad_t + h_offset * dilation_h;
        int w_pad = w * stride_w - pad_l + w_offset * dilation_w;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_col[(c * height_col + h) * width_col + w] =
              data_im[(c_im * height + h_pad) * width + w_pad];
        else
          data_col[(c * height_col + h) * width_col + w] = 0;
      }
    }
  }
}

void addBias(float *Y, const float *bias, const int filter_num, const int out_size)
{
  for (int i = 0; i < filter_num; ++i)
    for (int j = 0; j < out_size; ++j)
      Y[i * out_size + j] += bias[i];
}

void pool2D_f32_imp(float *X, int *X_shape, float *Y,
                    int *Y_shape, int size, int *pads, int *strides, bool ave)
{
  int batch_size = X_shape[0];
  int channels = X_shape[1];
  int height = X_shape[2];
  int width = X_shape[3];
  int pooled_height = Y_shape[2];
  int pooled_width = Y_shape[3];
  int stride_h = strides[0];
  int stride_w = strides[1];

  for (int n = 0; n < batch_size; ++n)
  {
    for (int c = 0; c < channels; ++c)
    {
      for (int ph = 0; ph < pooled_height; ++ph)
      {
        int hstart = ph * stride_h - pads[0];
        int hend = std::min(hstart + size, height);
        hstart = std::max(hstart, 0);
        for (int pw = 0; pw < pooled_width; ++pw)
        {
          int wstart = pw * stride_w - pads[1];
          int wend = std::min(wstart + size, width);
          wstart = std::max(wstart, 0);
          const int pool_index = ph * pooled_width + pw;
          float Yh = ave ? 0 : std::numeric_limits<float>::lowest();
          for (int h = hstart; h < hend; ++h)
          {
            for (int w = wstart; w < wend; ++w)
            {
              const int input_index = h * width + w;
              Yh = ave ? Yh + X[input_index] : (X[input_index] > Yh ? X[input_index] : Yh);
            }
          }
          Yh = ave ? Yh / (hend - hstart) * (wend - wstart) : Yh;
          Y[pool_index] = Yh;
        }
      }
      // Do offset.
      X += height * width;
      Y += pooled_height * pooled_width;
    }
  }
}

static inline float logistic_activate(float x)
{
  return 1.f / (1.f + expf(-x));
}
static inline float relu_activate(float x) { return x * (x > 0); }
static inline float leaky_activate(float x) { return (x > 0) ? x : 0.1 * x; }
static inline float tanh_activate(float x) { return (2 / (1 + expf(-2 * x)) - 1); }
static inline float softplus_activate(float x, float threshold)
{
  if (x > threshold)
    return x; // too large
  else if (x < -threshold)
    return expf(x); // too small
  return logf(expf(x) + 1.f);
}

static inline float mish_activate(float x)
{
  return x * tanh_activate(softplus_activate(x, 20));
}
static inline float swish_activate(float x)
{
  return x * logistic_activate(x);
}
float *activate(float *x, const int length, const int a)
{
  switch (a)
  {
  case 1:
    for (int i = 0; i < length; i++)
      x[i] = logistic_activate(x[i]);
    return x;
  case 2:
    for (int i = 0; i < length; i++)
      x[i] = relu_activate(x[i]);
    return x;
  case 3:
    for (int i = 0; i < length; i++)
      x[i] = leaky_activate(x[i]);
    return x;
  case 4:
    for (int i = 0; i < length; i++)
      x[i] = mish_activate(x[i]);
    return x;
  case 5:
    for (int i = 0; i < length; i++)
      x[i] = swish_activate(x[i]);
    return x;
  }
  return x;
}
void matmul(float *A, float *B, float *C, const int M, const int N, const int K)
{
  const int alignedN = N - N % 4;
  const int alignedM = M - M % 4;
  /*  for (int j = 0; j < alignedN; j += 8)
  {
    buf_t bufB(8 * K);
    reorder_b_8(K, B + j, N, bufB.p);
    for (int i = 0; i < alignedM; i += 4)
      micro4x8(K, A + i * K, K, bufB.p, 8, C + i * N + j, N);
    for (int i = alignedM; i < M; ++i)
      for (int k = 0; k < K; ++k)
        C[i * N + j] += A[i * K + k] * B[k * N + j];
  }

  for (int j = alignedN; j < N; ++j)
    for (int i = 0; i < M; ++i)
      for (int k = 0; k < K; ++k)
        C[i * N + j] += A[i * K + k] * B[k * N + j];*/
  for (int n = 0; n < alignedN; n += 4)
  {
    for (int m = 0; m < alignedM; m += 4)
    {
      float C00 = 0, C01 = 0, C02 = 0, C03 = 0, C10 = 0, C11 = 0, C12 = 0, C13 = 0, C20 = 0, C21 = 0, C22 = 0, C23 = 0, C30 = 0, C31 = 0, C32 = 0, C33 = 0;

      buf_t bufB(4 * K);
      //for (int k = 0; k < K; ++k)
      //  wasm_v128_store(bufB.p + k * 4 + 0, wasm_v128_load(B + k * N + 0));
      for (int k = 0; k < K; ++k)
      {
        bufB.p[4 * k + 0] = B[k * N + n + 0];
        bufB.p[4 * k + 1] = B[k * N + n + 1];
        bufB.p[4 * k + 2] = B[k * N + n + 2];
        bufB.p[4 * k + 3] = B[k * N + n + 3];
      }
      for (int k = 0; k < K; ++k)
      {
        C00 += A[(m + 0) * K + k] * bufB.p[k * 4 + 0];
        C10 += A[(m + 1) * K + k] * bufB.p[k * 4 + 0];
        C20 += A[(m + 2) * K + k] * bufB.p[k * 4 + 0];
        C30 += A[(m + 3) * K + k] * bufB.p[k * 4 + 0];

        C01 += A[(m + 0) * K + k] * bufB.p[k * 4 + 1];
        C11 += A[(m + 1) * K + k] * bufB.p[k * 4 + 1];
        C21 += A[(m + 2) * K + k] * bufB.p[k * 4 + 1];
        C31 += A[(m + 3) * K + k] * bufB.p[k * 4 + 1];

        C02 += A[(m + 0) * K + k] * bufB.p[k * 4 + 2];
        C12 += A[(m + 1) * K + k] * bufB.p[k * 4 + 2];
        C22 += A[(m + 2) * K + k] * bufB.p[k * 4 + 2];
        C32 += A[(m + 3) * K + k] * bufB.p[k * 4 + 2];

        C03 += A[(m + 0) * K + k] * bufB.p[k * 4 + 3];
        C13 += A[(m + 1) * K + k] * bufB.p[k * 4 + 3];
        C23 += A[(m + 2) * K + k] * bufB.p[k * 4 + 3];
        C33 += A[(m + 3) * K + k] * bufB.p[k * 4 + 3];
      }

      C[(m + 0) * N + n + 0] = C00, C[(m + 0) * N + n + 1] = C01, C[(m + 0) * N + n + 2] = C02, C[(m + 0) * N + n + 3] = C03;
      C[(m + 1) * N + n + 0] = C10, C[(m + 1) * N + n + 1] = C11, C[(m + 1) * N + n + 2] = C12, C[(m + 1) * N + n + 3] = C13;
      C[(m + 2) * N + n + 0] = C20, C[(m + 2) * N + n + 1] = C21, C[(m + 2) * N + n + 2] = C22, C[(m + 2) * N + n + 3] = C23;
      C[(m + 3) * N + n + 0] = C30, C[(m + 3) * N + n + 1] = C31, C[(m + 3) * N + n + 2] = C32, C[(m + 3) * N + n + 3] = C33;
    }
    for (int j = alignedM; j < M; ++j)
      for (int k = 0; k < K; ++k)
        C[j * N + n] += A[j * K + k] * B[k * N + n];
  }
  for (int i = alignedM; i < M; ++i)
    for (int k = 0; k < K; ++k)
      for (int j = 0; j < N; ++j)
        C[i * N + j] += A[i * K + k] * B[k * N + j];
}

void micro4x8(size_t K, const float *A, size_t lda, const float *B, size_t ldb, float *C, size_t ldc)
{
  v128_t c00 = wasm_f32x4_const(0, 0, 0, 0);
  v128_t c10 = wasm_f32x4_const(0, 0, 0, 0);
  v128_t c20 = wasm_f32x4_const(0, 0, 0, 0);
  v128_t c30 = wasm_f32x4_const(0, 0, 0, 0);
  v128_t c01 = wasm_f32x4_const(0, 0, 0, 0);
  v128_t c11 = wasm_f32x4_const(0, 0, 0, 0);
  v128_t c21 = wasm_f32x4_const(0, 0, 0, 0);
  v128_t c31 = wasm_f32x4_const(0, 0, 0, 0);
  v128_t b0, b1, a0;
  const size_t oa0 = lda * 0;
  const size_t oa1 = lda * 1;
  const size_t oa2 = lda * 2;
  const size_t oa3 = lda * 3;
  const size_t sa = lda == 1 ? 4 : 1;
  for (size_t k = 0; k < K; k++)
  {
    b0 = wasm_v128_load(B + 0);
    b1 = wasm_v128_load(B + 4);
    a0 = wasm_f32x4_splat(A[oa0]);
    c00 = wasm_f32x4_add(wasm_f32x4_mul(a0, b0), c00);
    c01 = wasm_f32x4_add(wasm_f32x4_mul(a0, b1), c01);
    a0 = wasm_f32x4_splat(A[oa1]);
    c10 = wasm_f32x4_add(wasm_f32x4_mul(a0, b0), c10);
    c11 = wasm_f32x4_add(wasm_f32x4_mul(a0, b1), c11);
    a0 = wasm_f32x4_splat(A[oa2]);
    c20 = wasm_f32x4_add(wasm_f32x4_mul(a0, b0), c20);
    c21 = wasm_f32x4_add(wasm_f32x4_mul(a0, b1), c21);
    a0 = wasm_f32x4_splat(A[oa3]);
    c30 = wasm_f32x4_add(wasm_f32x4_mul(a0, b0), c30);
    c31 = wasm_f32x4_add(wasm_f32x4_mul(a0, b1), c31);
    B += ldb;
    A += sa;
  }
  wasm_v128_store(C + 0, wasm_f32x4_add(c00, wasm_v128_load(C + 0)));
  wasm_v128_store(C + 4, wasm_f32x4_add(c01, wasm_v128_load(C + 4)));
  C += ldc;
  wasm_v128_store(C + 0, wasm_f32x4_add(c10, wasm_v128_load(C + 0)));
  wasm_v128_store(C + 4, wasm_f32x4_add(c11, wasm_v128_load(C + 4)));
  C += ldc;
  wasm_v128_store(C + 0, wasm_f32x4_add(c20, wasm_v128_load(C + 0)));
  wasm_v128_store(C + 4, wasm_f32x4_add(c21, wasm_v128_load(C + 4)));
  C += ldc;
  wasm_v128_store(C + 0, wasm_f32x4_add(c30, wasm_v128_load(C + 0)));
  wasm_v128_store(C + 4, wasm_f32x4_add(c31, wasm_v128_load(C + 4)));
}
void reorder_b_8(int K, const float *B, int ldb, float *bufB)
{
  for (int k = 0; k < K; ++k, B += ldb, bufB += 8)
  {
    wasm_v128_store(bufB + 0, wasm_v128_load(B + 0));
    wasm_v128_store(bufB + 4, wasm_v128_load(B + 4));
  }
}