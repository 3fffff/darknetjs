#pragma once
#include <vector>
#include <algorithm>
#include <limits>
#include <stdint.h>
#include <math.h>
#include <cmath>
#include <stdlib.h>
#include <mm_malloc.h>
#include <wasm_simd128.h>

extern "C"
{
  void conv_f32(void *);
  void convT_f32(void *);
  void pool_f32(void *);
  void matmul_f32(void *);

  void conv2D_f32_imp(float *, int32_t *, float *, int32_t *, float *, int32_t *, float *, int32_t *, int32_t, int32_t *,
                      int32_t *, int32_t, float *, float *, float *);
  void convTranspose2D_f32_imp(float *, int32_t *, float *, int32_t *, float *, int32_t *,
                               float *, int32_t *, int32_t, int32_t *, int32_t *, int32_t, float *, float *, float *);
  void im2col_f32(const float *, float *, const int32_t, const int32_t, const int32_t,
                  const int32_t, const int32_t, const int32_t, const int32_t,
                  const int32_t, const int32_t, const int32_t, const int32_t,
                  const int32_t, const int32_t);
  void connected_imp(float *, int *, float *, int *, float *, int *, float *, int, float *, float *, float *);
  void matmul(float *, float *, float *, const int32_t, const int32_t, const int32_t);
 /* void convdw3x3s1(float *const &, const int &, const int &, const int &, float *const &,
                   float *&, const int &, const int &, const int &, float *const &);
  void convdw3x3s2(float *const &, const int &, const int &, const int &, float *const &,
                   float *&, const int &, const int &, const int &, float *const &);*/
  void reorder_a_4(const float * , int , int , int , float *);
  void pool2D_f32_imp(float *, int *, float *, int *, int, int *, int *, bool);
  float *activate(float *, const int, const int);
  void add_bias(float *, const float *, const int32_t, const int32_t, float *, float *, float *);
  void micro4x8(size_t, const float *, size_t, const float *, size_t, float *, size_t);
  void reorder_b_8(int, const float *, int, float *);
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
                    int *pads, int *strides, int active, float *scales, float *mean, float *variance)
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
  const int filter_size = filter_num * filter_channels * filter_height * filter_width;

  const int output_num = Y_shape[0];
  const int output_channels = Y_shape[1];
  const int output_height = Y_shape[2];
  const int output_width = Y_shape[3];
  const int output_size = output_num * output_channels * output_height * output_width;

  const int output_image_size = output_height * output_width;
  const int X_offset = input_channels / groups * input_height * input_width;
  const int Y_offset = output_size / output_num / groups;
  const int W_offset = filter_size / groups;
  const int kernel_dim = input_channels / groups * filter_height * filter_width;
  const int col_buffer_size = kernel_dim * output_image_size;
  const int out_size = output_height * output_width;
  
  for (int n = 0; n < input_num; ++n)
  {
    /*if (strides[0] == 1 && strides[1] == 1 && groups != 1 && filter_height == 3 && filter_width == 3 && scales == nullptr)
    {
      convdw3x3s1(X, input_width, input_height, groups, W, Y, output_width, output_height, output_channels, bias);
      Y = activate(Y, output_size, active);
      return;
    }
    else 
    if (strides[0] == 2 && strides[1] == 2 && groups != 1 && filter_height == 3 && filter_width == 3 && scales == nullptr)
    {
      convdw3x3s2(X, input_width, input_height, groups, W, Y, output_width, output_height, output_channels, bias);
      Y = activate(Y, output_size, active);
      return;
    }
    else
    {*/
      float *col_buffer_data = new float[col_buffer_size];
      for (int group = 0; group < groups; ++group)
      {
        im2col_f32(X + X_offset * group, col_buffer_data, input_channels / groups, input_height,
                   input_width, filter_height, filter_width, dilations[0],
                   dilations[1], pads[0], pads[1], pads[2], pads[3], strides[0],
                   strides[1]);
        matmul(W + W_offset * group, col_buffer_data, Y + Y_offset * group, filter_num / groups, output_height * output_width, kernel_dim);
      }
      delete[] col_buffer_data;
      if (bias != nullptr)
        add_bias(Y, bias, filter_num, out_size, scales, mean, variance);
      Y = activate(Y, output_size, active);
   // }
  }
}

void convTranspose2D_f32_imp(float *X, int *X_shape, float *W, int *W_shape,
                             float *Y, int *Y_shape, float *bias, int *dilations,
                             int groups, int *pads, int *strides, int active, float *scales, float *mean, float *variance)
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
      im2col_f32(col_buffer_data, Y + Y_offset * group, output_channels / groups,
                 output_height, output_width, filter_height, filter_width,
                 dilations[0], dilations[1], pads[0], pads[1], pads[2], pads[3],
                 strides[0], strides[1]);
    }
  }
  if (bias != nullptr)
    add_bias(Y, bias, filter_num, out_size, scales, mean, variance);
  Y = activate(Y, output_size, active);
  delete[] col_buffer_data;
}

// Some helpers specific to conv operator
void im2col_f32(const float *data_im, float *data_col, const int channels, const int height,
                const int width, const int kernel_h, const int kernel_w,
                const int dilation_h, const int dilation_w, const int pad_t,
                const int pad_l, const int pad_b, const int pad_r,
                const int stride_h, const int stride_w)
{
  // Baseline
  const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int dkernel_w = dilation_w * (kernel_w - 1) + 1;

  const int height_col = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  const int width_col = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;

  const int channels_col = channels * kernel_h * kernel_w;
  for (int c = 0; c < channels_col; ++c)
  {
    const int w_offset = c % kernel_w;
    const int h_offset = (c / kernel_w) % kernel_h;
    const int c_im = c / (kernel_h * kernel_w);
    for (int h = 0; h < height_col; ++h)
    {
      for (int w = 0; w < width_col; ++w)
      {
        const int h_pad = h * stride_h - pad_t + h_offset * dilation_h;
        const int w_pad = w * stride_w - pad_l + w_offset * dilation_w;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_col[(c * height_col + h) * width_col + w] =
              data_im[(c_im * height + h_pad) * width + w_pad];
        else
          data_col[(c * height_col + h) * width_col + w] = 0;
      }
    }
  }
}
void add_bias(float *Y, const float *bias, const int filter_num, const int out_size, float *scales, float *mean, float *variance)
{
  for (int i = 0; i < filter_num; ++i)
    for (int j = 0; j < out_size; ++j)
      if (scales == nullptr)
        Y[i * out_size + j] += bias[i];
      else
        Y[i * out_size + i] = scales[i] * (Y[i * out_size + i] - mean[i]) / std::sqrt(variance[i] + 0.000001) + bias[i];
}
void connected_imp(float *X, int *X_shape, float *W, int *W_shape,
                   float *Y, int *Y_shape, float *bias, int active, float *scales, float *mean, float *variance)
{
  matmul(X, W, Y, X_shape[1], W_shape[1], Y_shape[1]);
  add_bias(Y, bias, W_shape[1], Y_shape[2] * Y_shape[3], scales, mean, variance);
  Y = activate(Y, Y_shape[2] * Y_shape[3], active);
}

void pool2D_f32_imp(float *X, int *X_shape, float *Y,
                    int *Y_shape, int size, int *pads, int *strides, int type)
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
          const int pool_index = ph * pooled_width + pw;
          if (type == 3) {
              for (let i = 0; i < height*width; ++i) Y[pool_index] += (X[ph + height*width * (c + n * channels)]) / (height*width);
          }else{ 
            int wstart = pw * stride_w - pads[1];
            int wend = std::min(wstart + size, width);
            wstart = std::max(wstart, 0);
            float Yh = type == 2 ? 0 : std::numeric_limits<float>::lowest();
            for (int h = hstart; h < hend; ++h)
            {
              for (int w = wstart; w < wend; ++w)
              {
                const int input_index = h * width + w;
                Yh = type == 2 ? Yh + X[input_index] : (X[input_index] > Yh ? X[input_index] : Yh);
              }
            }
            Yh = type == 2 ? Yh / (hend - hstart) * (wend - wstart) : Yh;
            Y[pool_index] = Yh;
          }
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

void macro(int M, int N, int K, const float * A, 
    const float * B, int ldb, float * bufB, bool reorderB, float * C, int ldc)
{
    for (int j = 0; j < N; j += 8)
    {
        if(reorderB)
            reorder_b_8(K, B + j, ldb, bufB + K*j);
        for (int i = 0; i < M; i += 4)
            micro_4x8(K, A + i*K, 1, 4, bufB + K*j, 8, C + i*ldc + j, ldc);
    }
}

void matmul(float *A, float *B, float *C, const int M, const int N, const int K)
{
  const int alignedN = N - N % 8;
  const int alignedM = M - M % 4;
  const int alignedK = K - K % 4;
  const int L1 = 32 * 1024, L2 = 256*1024, L3 = 2*1024*1024;
  int mK = std::min(L1 / 4 / 4, alignedK) / 4 * 4;
  int mM = std::min(L2 / 4 / mK, alignedM) / 4 * 4;
  int mN = std::min(L3 / 4 / mK, alignedN) / 8 * 8;
  buf_t bufB(mN * mK);
  buf_t bufA(mK * mM);
  for (int j = 0; j < alignedN; j += mN)
  {
    int dN = std::min(alignedN, j + mN) - j;
    for (int k = 0; k < alignedK; k += mK)
    {
      int dK = std::min(alignedK, k + mK) - k;
      for (int i = 0; i < alignedM; i += mM)
      {
        int dM = std::min(alignedM, i + mM) - i;
        reorder_a_4(A + i * K + k, K, dM, dK, bufA.p);
        macro(dM, dN, dK, bufA.p, B + k * N + j, N, 
                  bufB.p, i == 0, C + i * N + j, N);
      }
    }
  }
  for (int j = 0; j < N; ++j)
    for (int i = 0; i < M; ++i)
      for (int k = alignedK; k < K; ++k)
        C[i * N + j] += A[i * K + k] * B[k * N + j];
  for (int j = 0; j < N; ++j)
    for (int i = alignedM; i < M; ++i)
      for (int k = 0; k < K; ++k)
        C[i * N + j] += A[i * K + k] * B[k * N + j];
  for (int i = 0; i < M; ++i)
  {
    float *c = C + i * N;
    for (int k = 0; k < K; ++k)
    {
      const float *b = B + k * N;
      float a = A[i * K + k];
      for (int j = alignedN; j < N; ++j)
        c[j] += a * b[j];
    }
  }
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

void reorder_a_4(const float * A, int lda, int M, int K, float * bufA)
{
  for (int i = 0; i < M; i += 4)
  {
    for (int k = 0; k < K; k += 4)
    {
      const float * pA = A + k;
      v128_t s0 = wasm_v128_load(pA + 0 * stride);
      v128_t s1 = wasm_v128_load(pA + 1 * stride);
      v128_t s2 = wasm_v128_load(pA + 2 * stride);
      v128_t s3 = wasm_v128_load(pA + 3 * stride);
      const v128_t s00 = wasm_v32x4_shuffle(s0, s1, 0, 4, 1, 5);
	    const v128_t s01 = wasm_v32x4_shuffle(s2, s3, 0, 4, 1, 5);
	    const v128_t s10 = wasm_v32x4_shuffle(s0, s1, 2, 6, 3, 7);
	    const v128_t s11 = wasm_v32x4_shuffle(s2, s3, 2, 6, 3, 7);
      wasm_v128_store(bufA + 0, wasm_v32x4_shuffle(s00, s01, 0, 1, 4, 5));
      wasm_v128_store(bufA + 4, wasm_v32x4_shuffle(s00, s01, 2, 3, 6, 7));
      wasm_v128_store(bufA + 8, wasm_v32x4_shuffle(s10, s11, 0, 1, 4, 5));
      wasm_v128_store(bufA + 12, wasm_v32x4_shuffle(s10, s11, 2, 3, 6, 7));
      bufA += 16;
    }
    A += 4 * lda;
  }
}