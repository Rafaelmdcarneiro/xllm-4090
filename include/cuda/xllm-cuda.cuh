#pragma once
#include "../../include/data.h"

void *xllmCudaMalloc(size_t size);
void xllmCudaFree(void *ret);
void xllmCudaClearBigBuffer();
void xllmCudaCopyFromHostToDevice(void *dst, void *src, size_t size);
void xllmCudaCopyFromDeviceToHost(void *dst, void *src, size_t size);
void xllmCudaCopyFromDeviceToDevice(void *dst, void *src, size_t size);
void xllmCudaCopyFromDeviceToDeviceStream(int n, int strideBytes, void *dst, void *src, 
        std::vector<int> remainBatch);
void xllmCudaCopyFromDeviceToDevice(int n, int strideBytes, void *dst, void *src, 
        std::vector<int> remainBatch);
                
void *xllmCudaPrepareInput(const xllm::Data &input);
void xllmCudaFinishInput(const xllm::Data &input, void *data);
void xllmCudaSetDevice(int gpu_id);
void xllmCudaMemcpy2DDeviceToDevice(void * 	dst, size_t 	dpitch, const void * 	src,
                                       size_t 	spitch, size_t 	width, size_t 	height);
void xllmCudaMemcpy2DDeviceToDeviceFP16(void * 	dst, size_t 	dpitch, const void * 	src,
                                       size_t 	spitch, size_t 	width, size_t 	height);

bool xllmCudaRMSNorm(const xllm::Data &input, xllm::Data &weight, xllm::Data &output, float eps);
bool xllmCudaMatMulFloat16(const xllm::Data &input, xllm::Data &weight, const xllm::Data &bias, xllm::Data &output, int n, int m, int k);
bool xllmCudaLlamaRotatePosition2D(xllm::Data &data, const xllm::Data &positionIds,
                                      const xllm::Data &sinData, const xllm::Data &cosData, int rotaryDim);
bool xllmCudaPermute(xllm::Data &input, const std::vector<int> &axis);
bool xllmCudaBatchMatMulTransB(const xllm::Data &input0, const xllm::Data &input1, xllm::Data &output,
                              int input0Spatial, int input1Spatial, int outputSpatial,
                              int input0Stride, int input1Stride,
                              int batch, int n, int m, int k, float alpha);
bool xllmCudaBatchMatMulTransBFP16(const xllm::Data &input0, const xllm::Data &input1, xllm::Data &output,
                              int input0Spatial, int input1Spatial, int outputSpatial,
                              int input0Stride, int input1Stride,
                              int batch, int n, int m, int k, float alpha);
bool xllmCudaAttentionMask(xllm::Data &input, const xllm::Data &mask, float maskValue);
bool xllmCudaSoftmax(const xllm::Data &input, xllm::Data &output, int axis);
bool xllmCudaBatchMatMul(const xllm::Data &input0, const xllm::Data &input1, xllm::Data &output,
                                  int input0Spatial, int input1Spatial, int outputSpatial,
                                  int input0Stride, int input1Stride,
                                  int batch, int n, int m, int k, float alpha);
bool xllmCudaBatchMatMulFP16(const xllm::Data &input0, const xllm::Data &input1, xllm::Data &output,
                                  int input0Spatial, int input1Spatial, int outputSpatial,
                                  int input0Stride, int input1Stride,
                                  int batch, int n, int m, int k, float alpha);
bool xllmCudaAddTo(xllm::Data &input0, const xllm::Data &input1, float alpha);

bool xllmCudaSilu(const xllm::Data &input, xllm::Data &output);
bool xllmCudaMulTo(xllm::Data &input0, const xllm::Data &input1, float alpha);
bool xllmCudaTopK(const xllm::Data &input, xllm::Data &output, int topk);
bool xllmCudaAttention(const xllm::Data &q, const xllm::Data &k, const xllm::Data &v,
                          const xllm::Data &mask, const xllm::Data &output, int group, float scale);
