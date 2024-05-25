#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <map>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "nvtx3/nvToolsExt.h" 

#include "../../include/cuda/xllm-cuda.cuh"

// #define CHECK(call)
// {
//     const cudaError_t error = call;
//     if (error != cudaSuccess)
//     { 
//         printf("Error: %s:%d, ", __FILE__, __LINE__);
//         printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));
//         exit(1);
//     }
// }

static std::map<int, cublasHandle_t> s_xllmCublasHandleMap;
cublasHandle_t getxllmCublasHandle() {
    int id = -1;
    cudaGetDevice(&id);
    auto it = s_xllmCublasHandleMap.find(id);
    if (it != s_xllmCublasHandleMap.end()) {
        return it->second;
    }
    cublasHandle_t handler = nullptr;
    auto stat = cublasCreate(&handler);
    // cudaDeviceSynchronize();

    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed:%d\n", stat);
        exit(0);
    } else {
        s_xllmCublasHandleMap[id] = handler;
    }

    return handler;
}


void xllmCudaSetDevice(int gpu_id) {
    cudaSetDevice(gpu_id);
}

struct CudaMemoryBuffer {
    void *data;
    size_t size;
    bool busy;

    CudaMemoryBuffer () {}

    CudaMemoryBuffer (void *data, size_t size, bool busy) :
            data(data), size(size), busy(busy) {}
};

std::map<int, std::vector <CudaMemoryBuffer>> cudaBuffersMap;
std::map<int, std::vector <CudaMemoryBuffer>> bigBuffersMap;

void * xllmCudaMalloc(size_t size) {
    int id = -1;
    cudaGetDevice(&id);
    if (size > 1024 * 1024) {
        auto &bigBuffers = bigBuffersMap[id];
        int selId = -1;
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (bigBuffers[i].size >= size && !bigBuffers[i].busy
                && bigBuffers[i].size - size < 8 * 1024 * 1024) {
                if (selId == -1 || bigBuffers[selId].size > bigBuffers[i].size) {
                    selId = i;
                }
            }
        }
        if (selId != -1) {
            bigBuffers[selId].busy = true;
            return bigBuffers[selId].data;
        }

        void * ret;
        cudaMalloc(&ret, size);
        // cudaError_t cudaStatus = cudaDeviceSynchronize();
        // if (cudaStatus != cudaSuccess) {
        //     fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        // }
        bigBuffers.push_back(CudaMemoryBuffer(ret, size, true));
        return ret;
    }
    auto &cudaBuffers = cudaBuffersMap[id];
    for (int i = 0; i < cudaBuffers.size(); i++) {
        if (cudaBuffers[i].size >= size && !cudaBuffers[i].busy) {
            cudaBuffers[i].busy = true;
            return cudaBuffers[i].data;
        }
    }
    void * ret;
    cudaMalloc(&ret, size);
    // cudaError_t cudaStatus = cudaDeviceSynchronize();
    // if (cudaStatus != cudaSuccess) {
    //     fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
    // }
    cudaBuffers.push_back(CudaMemoryBuffer(ret, size, true));
    return ret;
}

void xllmCudaFree(void *ret) {
    if (ret == nullptr) {
        return;
    }
    for (auto &it : cudaBuffersMap) {
        auto &cudaBuffers = it.second;
        for (int i = 0; i < cudaBuffers.size(); i++) {
            if (cudaBuffers[i].data == ret) {
                cudaBuffers[i].busy = false;
                return;
            }
        }
        auto &bigBuffers = bigBuffersMap[it.first];
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (bigBuffers[i].data == ret) {
                bigBuffers[i].busy = false;
                return;
            }
        }
    }
    cudaFree(ret);
}

void xllmCudaClearBigBuffer() {
    int id = -1;
    cudaGetDevice(&id);
    for (auto &it : bigBuffersMap) {
        auto &bigBuffers = it.second;
        std::vector <CudaMemoryBuffer> temp;
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (!bigBuffers[i].busy) {
                cudaSetDevice(it.first);
                cudaFree(bigBuffers[i].data);
            } else {
                temp.push_back(bigBuffers[i]);
            }
        }
        bigBuffers.clear();
        bigBuffers = temp;
    }
    cudaSetDevice(id);
}


void xllmCudaCopyFromHostToDevice(void *dst, void *src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

void xllmCudaCopyFromDeviceToHost(void *dst, void *src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

void xllmCudaCopyFromDeviceToDevice(void *dst, void *src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
}

void xllmCudaCopyFromDeviceToDevice(int n, int strideBytes, void *dst, void *src, 
        std::vector<int> remainBatch) {

    for (int i = 0; i < n; i++){
        cudaMemcpy((uint8_t*)dst + i*strideBytes, 
        (uint8_t*)src + remainBatch[i]*strideBytes, strideBytes, cudaMemcpyDeviceToDevice);
    }
}

void xllmCudaCopyFromDeviceToDeviceStream(int n, int strideBytes, void *dst, void *src, 
        std::vector<int> remainBatch) {
    std::vector<cudaStream_t> streams(n);
    for (int i = 0; i < n; ++i) {
        cudaStreamCreate(&streams[i]);
    }
    for (int i = 0; i < n; i++){
        cudaMemcpyAsync((uint8_t*)dst + i*strideBytes, 
        (uint8_t*)src + remainBatch[i]*strideBytes, strideBytes, cudaMemcpyDeviceToDevice, streams[i]);
    }
    for (int i = 0; i < n; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    for (int i = 0; i < n; ++i) {
        cudaStreamDestroy(streams[i]);
    }
}

void *xllmCudaPrepareInput(const xllm::Data &input) {
    void *ret;
    if (input.dataDevice == xllm::DataDevice::CUDA) {
        ret = (void*)input.cudaData;
    } else {
        ret = (void*)(input.assignBytes);
        cudaMemcpy(ret, input.cpuData, input.assignBytes, cudaMemcpyHostToDevice);
    }
    return ret;
}

void xllmCudaFinishInput(const xllm::Data &input, void *data) {
    if (input.dataDevice != xllm::DataDevice::CUDA) {
        xllmCudaFree(data);
    }
}

void *xllmCudaPrepareOutput(xllm::Data &output) {
    void *ret;
    if (output.dataDevice == xllm::DataDevice::CUDA) {
        ret = (float*)output.cudaData;
    } else {
        ret = (float*)xllmCudaMalloc(output.assignBytes);
    }
    return ret;
}

void xllmCudaFinishOutput(xllm::Data &output, void *data) {
    if (output.dataDevice != xllm::DataDevice::CUDA) {
        cudaMemcpy(output.cpuData, data, output.assignBytes, cudaMemcpyDeviceToHost);
        xllmCudaFree(data);
    }
    // cudaDeviceSynchronize();
}

__global__ void xllmCudaFloat2HalfKernel(float* a, half *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        b[idx] = __float2half(a[idx]);
    }
}

__global__ void xllmCudaHalf2FloatKernel(half* a, float *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        b[idx] = __half2float(a[idx]);
    }
}

void xllmCudaMemcpy2DDeviceToDeviceFP16(void * 	dst, size_t 	dpitch, const void * 	src,
                                       size_t 	spitch, size_t 	width, size_t 	height) {
    int len = width * height / 2;
    int threadPerBlock = std::min(256, len);
    half *cudaFp16Input = (half *) xllmCudaMalloc(len * 2);
    xllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((float*) src, cudaFp16Input,
                                                                                        len);                                   
    // cudaError_t status = cudaDeviceSynchronize();
    // if (status != cudaSuccess)
    // { 
    //     printf("Error: %s:%d, ", __FILE__, __LINE__);
    //     printf("code:%d, reason: %s\n", status, cudaGetErrorString(status));
    //     exit(1);
    // }

    cudaMemcpy2D(dst, dpitch, cudaFp16Input, spitch, width, height, cudaMemcpyDeviceToDevice);
    // status = cudaDeviceSynchronize();
    // if (status != cudaSuccess)
    // { 
    //     printf("Error: %s:%d, ", __FILE__, __LINE__);
    //     printf("code:%d, reason: %s\n", status, cudaGetErrorString(status));
    //     exit(1);
    // }
    xllmCudaFree(cudaFp16Input);
}

void xllmCudaMemcpy2DDeviceToDevice(void * 	dst, size_t 	dpitch, const void * 	src,
                                       size_t 	spitch, size_t 	width, size_t 	height) {
    // Copy a matrix
    cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice);
    // cudaDeviceSynchronize();
}

template <int THREAD_PER_BLOCK>
__global__ void xllmRMSNormKernelInner1(float *input, float *weight, float *output, int outer, int channels, float eps) {
    int o = blockIdx.x;
    input = input + o * channels;
    output = output + o * channels;

    // 1. 计算平方和：每个线程计算一部分
    __shared__ float sdata2[THREAD_PER_BLOCK];  // share memory
    unsigned int tid = threadIdx.x;
    float sum2 = 0.0;
#pragma unroll
    for (int i = tid; i < channels; i += blockDim.x) {
        float x = input[i];
        sum2 += x * x;
    }
    sdata2[tid] = sum2;
    __syncthreads();  // sync within block

    // 2. sdata2[0] = sum(sdata2)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata2[tid] += sdata2[tid + s];
        }
        __syncthreads();
    }

    // 3. 计算参数
    __shared__ float scale;
    if (tid == 0) {
        scale = 1.0 / sqrt(sdata2[0] / channels + eps);
    }
    __syncthreads();

#pragma unroll
    for (int i = tid; i < channels; i += blockDim.x) {
        output[i] = (input[i] * scale * weight[i]);
    }
}

bool xllmCudaRMSNorm(const xllm::Data &input, xllm::Data &weight, xllm::Data &output, float eps) {
    nvtxRangePush("xllmCudaRMSNorm");
    float *cudaInput = (float *) xllmCudaPrepareInput(input);
    float *cudaOutput = (float *) xllmCudaPrepareInput(output);

    int dimsLen = input.dims.size();
    int axis = dimsLen - 1;
    int outer = input.Count(0) / input.Count(axis);
    int channels = input.dims[axis];

    if (channels < 64) {
        xllmRMSNormKernelInner1<1> <<< outer, 1 >>>(cudaInput, (float *) weight.cudaData, cudaOutput, outer, channels, eps);
    } else if (channels < 512) {
        xllmRMSNormKernelInner1<64> <<< outer, 64 >>>(cudaInput, (float *) weight.cudaData, cudaOutput, outer, channels, eps);
    } else {
        xllmRMSNormKernelInner1<512> <<< outer, 512 >>>(cudaInput, (float *) weight.cudaData, cudaOutput, outer, channels, eps);
    }

    xllmCudaFinishInput(input, cudaInput);
    xllmCudaFinishOutput(output, cudaOutput);
    nvtxRangePop();
    return true;
}

__global__ void xllmCudaBiasKernel(float *a, float *bias, int k) {
    float *now = a + blockIdx.x * k;
    int stride = blockDim.x;
    for (int i = threadIdx.x; i < k; i += stride) {
        now[i] += bias[i];
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void xllmGemvFp32Fp16Kernel2(float *A, half *B, float *C, float *bias, int m, int k) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 计算
    int st = blockIdx.x * PART;
    int end = st + PART;
    for (int p = st; p < end; p++) {
        sdata[tid] = 0;
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
            sdata[tid] += A[i] * (float)B[p * m + i];
        }
        __syncthreads();
        for (unsigned int s = 1; s < THREAD_PER_BLOCK; s *= 2) {
            if ((tid & (2 * s - 1)) == 0) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            C[p] = sdata[0] + bias[p];
        }
        __syncthreads();
    }
}

bool xllmCudaMatMulFloat16(const xllm::Data &input, xllm::Data &weight, const xllm::Data &bias, xllm::Data &output, int n, int m, int k) {
    nvtxRangePush("xllmCudaMatMulFloat16");
    if (weight.cudaData == nullptr || weight.extraCudaData.size() == 0) {
        float *cudaBiasData;
        cudaMalloc(&cudaBiasData, k * sizeof(float));
        if (bias.dims.size() > 0) {
            cudaMemcpy(cudaBiasData, (uint8_t*)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            cudaMemset(cudaBiasData, 0, k * sizeof(float));
        }
        weight.extraCudaData.push_back((void*)cudaBiasData);
    }
    float *cudaBiasData = (float*)weight.extraCudaData[0];
    float *cudaInput = (float*)xllmCudaPrepareInput(input);
    float *cudaOutput = (float*)xllmCudaPrepareOutput(output);

    if (n > 1) {
        // 矩阵*矩阵
        half *cudaFp16Input, *cudaFp16Output;
        cudaFp16Input = (half *) xllmCudaMalloc(n * m * sizeof(half));
        cudaFp16Output = (half *) xllmCudaMalloc(n * k * sizeof(half));

        __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
        auto fastllmCublasHandle = getxllmCublasHandle();
        // cudaDeviceSynchronize();
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
        cublasStatus_t status;

        int len = n * m;
        int threadPerBlock = std::min(256, len);
        xllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaFp16Input,
                                                                                          len);
        // cublasGemmEx is an extension of cublas<t>gemm that allows the user to individually specify the data types 
        // for each of the A, B and C matrices
        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              k, n, m,
                              &h_alpha, (half *) weight.cudaData, AType,
                              m, cudaFp16Input, BType,
                              m, &h_beta,
                              cudaFp16Output, CType,
                              k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            exit(0);
        }

        len = n * k;
        xllmCudaHalf2FloatKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp16Output, cudaOutput,
                                                                                           len);
        // xllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, (float*)weight.extraCudaData[0], k);
        // cudaDeviceSynchronize();

        xllmCudaFree(cudaFp16Input);
        xllmCudaFree(cudaFp16Output);
    } else {
        // 向量*矩阵
        xllmGemvFp32Fp16Kernel2<256, 1> <<< k, 256 >>>(cudaInput, (half *) weight.cudaData, cudaOutput, cudaBiasData, m, k);
    }

    xllmCudaFinishInput(input, cudaInput);
    xllmCudaFinishOutput(output, cudaOutput);
    // cudaDeviceSynchronize();
    nvtxRangePop();
    return true;
}

__global__ void xllmLlamaRotatePosition2DKernel(float *data, float *positionIds, float *sin, float *cos,
                                                   int len, int bs, int spatial, int n, int m, int partStride, int sinCosStride, int rotateDim) {
    int o = (blockIdx.x / n);
    int l = o % len;
    int b = o / len;
    int j = threadIdx.x;
    int index = (int) (positionIds[b * partStride + l]);

    float curSin = sin[index * sinCosStride + j];
    float curCos = cos[index * sinCosStride + j];
    float *d = (float *) data + o * spatial + j;
    int i = blockIdx.x % n;
    float va = d[i * m], vb = d[i * m + m / 2];
    d[i * m] = va * curCos - vb * curSin;
    d[i * m + m / 2] = va * curSin + vb * curCos;
}


bool xllmCudaLlamaRotatePosition2D(xllm::Data &data, const xllm::Data &positionIds,
                                      const xllm::Data &sinData, const xllm::Data &cosData, int rotaryDim) {
    float *cudaData = (float *) xllmCudaPrepareInput(data);
    float *cudaPositionIds = (float *) xllmCudaPrepareInput(positionIds);
    float *cudaSin = (float *) xllmCudaPrepareInput(sinData);
    float *cudaCos = (float *) xllmCudaPrepareInput(cosData);

    int outer = data.dims[0] * data.dims[1];
    int spatial = data.Count(2);
    int bs = data.dims[0], len = data.dims[1];
    int n = data.dims[2], m = data.dims[3];
    xllmLlamaRotatePosition2DKernel <<< outer * n, std::min(rotaryDim, m / 2) >>> (cudaData, cudaPositionIds, cudaSin, cudaCos,
                                                                                 len, bs, spatial, n, m,
                                                                                 (int)positionIds.dims.back(), (int)sinData.dims[1], rotaryDim);

    xllmCudaFinishInput(positionIds, cudaPositionIds);
    xllmCudaFinishInput(sinData, cudaSin);
    xllmCudaFinishInput(cosData, cudaCos);
    xllmCudaFinishOutput(data, cudaData);
    return true;
}

template <int THREAD_PER_BLOCK>
__global__ void xllmLayerNormKernelTop1(float *input, float *output, int channels) {
    __shared__ float idData[THREAD_PER_BLOCK];
    __shared__ float maxData[THREAD_PER_BLOCK];
    float *inputData = input + blockIdx.x * channels;
    float *outputData = output + blockIdx.x * 2;
    int tid = threadIdx.x;
    maxData[tid] = -1e100;
    for (int j = tid; j < channels; j += THREAD_PER_BLOCK) {
        if (inputData[j] > maxData[tid]) {
            maxData[tid] = inputData[j];
            idData[tid] = j;
        }
    }
    __syncthreads();

    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (maxData[tid] < maxData[tid + s]) {
                maxData[tid] = maxData[tid + s];
                idData[tid] = idData[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        outputData[0] = idData[0];
        outputData[1] = maxData[0];
    }
}

bool xllmCudaTopK(const xllm::Data &input, xllm::Data &output, int topk) {
    if (topk != 1) {
        printf("topk: unsupport topk > 1.");
        exit(0);
    }

    float *cudaInput = (float *) xllmCudaPrepareInput(input);
    float *cudaOutput = (float *) xllmCudaPrepareInput(output);

    int dimsLen = input.dims.size();
    int outer = input.Count(0) / input.Count(dimsLen - 1);
    int channels = input.dims[dimsLen - 1];

    xllmLayerNormKernelTop1 <256> <<< outer, 256 >>> (cudaInput, cudaOutput, channels);
    xllmCudaFinishInput(input, cudaInput);
    xllmCudaFinishOutput(output, cudaOutput);
    return true;
}

template <int THREAD_PER_BLOCK>
__global__ void xllmTransposeByRowKernel(uint8_t *dst, uint8_t *ori, int n, int m, int k) {
    int row = blockIdx.x / m, col = blockIdx.x % m;
    uint8_t *curInput = ori + (row * m + col) * k;
    uint8_t *curOutput = dst + (col * n + row) * k;
    for (int i = threadIdx.x; i < k; i += THREAD_PER_BLOCK) {
        curOutput[i] = curInput[i];
    }
}

__global__ void xllmPermuteKernel(float *dst, float *ori, int *temp, int axisLen, int len) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len) {
        int old = 0;
        int idx = i;
        for (int j = 0; j < axisLen; ++j) {
            int order = temp[j];
            old += (idx / temp[j + 2 * axisLen]) * temp[order + 1 * axisLen];
            idx %= temp[j + 2 * axisLen];
        }
        dst[i] = ori[old];
    }
}

bool xllmCudaPermute(xllm::Data &input, const std::vector<int> &axis) {
    if (input.dataDevice != xllm::DataDevice::CUDA) {
        printf("permute: data should in cuda.\n");
        exit(0);
    }
    int len = input.Count(0);
    float *tempData = (float *)xllmCudaMalloc(len * sizeof(float));
    cudaMemcpy(tempData, input.cudaData, len * sizeof(float), cudaMemcpyDeviceToDevice);

    std::vector<int> new_dims;
    for (int i = 0; i < axis.size(); i++) {
        new_dims.push_back(input.dims[axis[i]]);
    }
    if (axis == std::vector <int> {1, 0, 2}) {
        int n = input.dims[0];
        int m = input.dims[1];
        int k = input.dims[2];
        xllmTransposeByRowKernel <256> <<< n * m, 256 >>>
                ((uint8_t*)input.cudaData, (uint8_t*)tempData, n, m, k * input.unitSize);
        input.Resize(new_dims);
    } else if (axis == std::vector <int> {2, 0, 1, 3}) {
        int n = input.dims[0] * input.dims[1];
        int m = input.dims[2];
        int k = input.dims[3];
        xllmTransposeByRowKernel <256> <<< n * m, 256 >>>
                ((uint8_t*)input.cudaData, (uint8_t*)tempData, n, m, k * input.unitSize);
        input.Resize(new_dims);
    } else {
        std::vector<int> temp;
        int len = input.Count(0);
        for (int i = 0; i < axis.size(); i++) {
            temp.push_back(axis[i]);
        }
        for (int i = 0; i < axis.size(); i++) {
            temp.push_back(input.Count(i + 1));
        }
        input.Resize(new_dims);
        for (int i = 0; i < axis.size(); i++) {
            temp.push_back(input.Count(i + 1));
        }

        int *cudaTemp = (int *) xllmCudaMalloc(temp.size() * sizeof(int));
        cudaMemcpy(cudaTemp, temp.data(), temp.size() * sizeof(int), cudaMemcpyHostToDevice);
        int threadPerBlock = std::min(256, len);
        xllmPermuteKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>((float *) input.cudaData,
                                                                                    tempData, cudaTemp,
                                                                                    (int) axis.size(), len);
        xllmCudaFree(cudaTemp);
    }

    xllmCudaFree(tempData);
    return true;
}

bool xllmCudaBatchMatMulTransB(const xllm::Data &input0, const xllm::Data &input1, xllm::Data &output,
                                  int input0Spatial, int input1Spatial, int outputSpatial,
                                  int input0Stride, int input1Stride,
                                  int batch, int n, int m, int k, float alpha) {
    float *cudaInput0 = (float *) xllmCudaPrepareInput(input0);
    float *cudaInput1 = (float *) xllmCudaPrepareInput(input1);
    float *cudaOutput = (float *) xllmCudaPrepareOutput(output);
    float beta = 0;
    auto xllmCublasHandle = getxllmCublasHandle();
    cublasStatus_t status;

    // for float
    status = cublasSgemmStridedBatched(xllmCublasHandle,
                                       CUBLAS_OP_T, CUBLAS_OP_N,
                                       k, n, m, &alpha,
                                       cudaInput1, input1Stride, input1Spatial,
                                       cudaInput0, input0Stride, input0Spatial,
                                       &beta,
                                       cudaOutput, k, k * n, batch);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("status = %d\n", (int)status);
        printf("%d %d %d\n", k, n, m);
        printf("Error: cublas error.\n");
        throw("cublas error");
        exit(0);
    }

    xllmCudaFinishInput(input0, cudaInput0);
    xllmCudaFinishInput(input1, cudaInput1);
    xllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool xllmCudaBatchMatMulTransBFP16(const xllm::Data &input0, const xllm::Data &input1, xllm::Data &output,
                                  int input0Spatial, int input1Spatial, int outputSpatial,
                                  int input0Stride, int input1Stride,
                                  int batch, int n, int m, int k, float alpha) {
    nvtxRangePush("MatMulTransBFP16");
    float *cudaInput0 = (float *) xllmCudaPrepareInput(input0);
    float *cudaOutput = (float *) xllmCudaPrepareOutput(output);
    float beta = 0;
    half *cudaInput1FP16 = (half *) xllmCudaPrepareInput(input1);

    half* cudaInput0FP16 = (half*) xllmCudaMalloc(input0.counts * sizeof(half));
    half* cudaOutputFP16 = (half*) xllmCudaMalloc(output.counts * sizeof(half));
    int threadPerBlock = std::min(256, input0.counts);
    xllmCudaFloat2HalfKernel <<< (input0.counts - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaInput0, cudaInput0FP16, input0.counts);

    auto xllmCublasHandle = getxllmCublasHandle();
    cublasStatus_t status;

    __half h_alpha = __float2half_rn(alpha), h_beta = __float2half_rn(beta);
    status = cublasHgemmStridedBatched(xllmCublasHandle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    k, n, m, &h_alpha,
                                    cudaInput1FP16, input1Stride, input1Spatial,
                                    cudaInput0FP16, input0Stride, input0Spatial,
                                    &h_beta,
                                    cudaOutputFP16, k, k * n, batch);                           
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("status = %d\n", (int)status);
        printf("%d %d %d\n", k, n, m);
        printf("Error: cublas error.\n");
        throw("cublas error");
        exit(0);
    }

    threadPerBlock = std::min(256, output.counts);
    xllmCudaHalf2FloatKernel <<< (output.counts - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaOutputFP16, cudaOutput, output.counts);

    xllmCudaFree(cudaInput0FP16);
    xllmCudaFree(cudaOutputFP16);
    xllmCudaFinishInput(input0, cudaInput0);
    xllmCudaFinishInput(input1, cudaInput1FP16);
    xllmCudaFinishOutput(output, cudaOutput);
    
    nvtxRangePop();
    return true;
}

template <int THREAD_PER_BLOCK>
__global__ void xllmAttentionMaskKernel(float* a, float *b, float maskValue, int n, int m, int spatial) {
    int on = blockIdx.x / m;
    int om = blockIdx.x % m;
    int o = on * m + om;
    int idx = threadIdx.x;
    for (int i = idx; i < spatial; i += THREAD_PER_BLOCK) {
        if (b[on * spatial + i] > 0.99) {
            a[o * spatial + i] = maskValue;
        }
    }
}

bool xllmCudaAttentionMask(xllm::Data &input, const xllm::Data &mask, float maskValue) {
    int spatial = input.Count(2), n = input.dims[0], m = input.dims[1];
    float *cudaData = (float *) xllmCudaPrepareInput(input);
    float *maskData = (float *) xllmCudaPrepareInput(mask);

    xllmAttentionMaskKernel <256> <<< n * m, 256>>>(cudaData, maskData, maskValue,
                                                       n, m, spatial);
    xllmCudaFinishInput(mask, maskData);
    xllmCudaFinishOutput(input, cudaData);
    return true;
}

template <int THREAD_PER_BLOCK>
__device__ void xllmSoftmaxKernelInner1Func(const float * __restrict input, float * __restrict output, int channels) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float sdata2[THREAD_PER_BLOCK];

    // 1. 求max
    unsigned int tid = threadIdx.x;
    unsigned int len = (channels + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK; // 每个线程计算的数据数量
    float maxValue = input[tid];
    float maxValue_before;
    float sum = 0;
    for (int i = 0; i < len; i++) {
        if (tid + i * THREAD_PER_BLOCK < channels){
            maxValue_before = maxValue;
            maxValue = max(maxValue, input[tid + i * THREAD_PER_BLOCK]);
            sum = sum * exp(maxValue_before-maxValue) + exp(input[tid + i * THREAD_PER_BLOCK] - maxValue);
        }
    }
    sdata[tid] = maxValue;
    sdata2[tid] = sum;
    __syncthreads();

    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            maxValue_before = sdata[tid];
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
            sdata2[tid] = sdata2[tid] * exp(maxValue_before-sdata[tid]) + sdata2[tid + s] * exp(sdata[tid + s] - sdata[tid]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (fabs(sdata2[0]) < 1e-6) {
            sdata2[0] = 0.1;
        }
    }
    __syncthreads();

    // 2. 计算最终结果
    for (int i = 0; i < len; i++) {
        if (tid + i * THREAD_PER_BLOCK < channels)
            output[tid + i * THREAD_PER_BLOCK] = __fdividef(__expf(input[tid + i * THREAD_PER_BLOCK] - sdata[0]), sdata2[0]);
    }
}

// callable from the host
// 一个block处理一个outer的数据
template <int THREAD_PER_BLOCK>
__global__ void xllmSoftmaxKernelInner1(float* input, float *output, int outer, int channels) {
    int o = blockIdx.x;
    xllmSoftmaxKernelInner1Func <THREAD_PER_BLOCK> (input + o * channels, output + o * channels, channels);
}

bool xllmCudaSoftmax(const xllm::Data &input, xllm::Data &output, int axis) {
    float *cudaInput = (float *) xllmCudaPrepareInput(input);
    float *cudaOutput = (float *) xllmCudaPrepareInput(output);

    // float* hostData = (float*)malloc(input.assignBytes);
    // cudaMemcpy(hostData, cudaInput, input.assignBytes, cudaMemcpyDeviceToHost);

    int dimsLen = input.dims.size();
    axis = (axis % dimsLen + dimsLen) % dimsLen;
    int outer = input.Count(0) / input.Count(axis);
    int channels = input.dims[axis];
    int inner = input.Count(axis + 1);

    if (inner == 1) {
        if (channels < 8) {
            xllmSoftmaxKernelInner1 <1> <<< outer, 1 >>> (cudaInput, cudaOutput, outer, channels);
        } else if (channels < 64) {
            xllmSoftmaxKernelInner1 <16> <<< outer, 16 >>> (cudaInput, cudaOutput, outer, channels);
        } else if (channels < 512) {
            xllmSoftmaxKernelInner1 <64> <<< outer, 64 >>> (cudaInput, cudaOutput, outer, channels);
        } else {
            xllmSoftmaxKernelInner1 <256> <<< outer, 256 >>> (cudaInput, cudaOutput, outer, channels);
        }

    } else {
        printf("softmax error.\n");
        exit(0);
    }

    // float* hostDataOut = (float*)malloc(input.assignBytes);
    // cudaMemcpy(hostDataOut, cudaOutput, input.assignBytes, cudaMemcpyDeviceToHost);

    xllmCudaFinishInput(input, cudaInput);
    xllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool xllmCudaBatchMatMul(const xllm::Data &input0, const xllm::Data &input1, xllm::Data &output,
                            int input0Spatial, int input1Spatial, int outputSpatial,
                            int input0Stride, int input1Stride,
                            int batch, int n, int m, int k, float alpha) {
    float *cudaInput0 = (float *) xllmCudaPrepareInput(input0);
    float *cudaInput1 = (float *) xllmCudaPrepareInput(input1);
    float *cudaOutput = (float *) xllmCudaPrepareOutput(output);
    float beta = 0;
    auto xllmCublasHandle = getxllmCublasHandle();
    cublasStatus_t status;

    status = cublasSgemmStridedBatched(xllmCublasHandle,
                                       CUBLAS_OP_N, CUBLAS_OP_N,
                                       k, n, m, &alpha,
                                       cudaInput1, input1Stride, input1Spatial,
                                       cudaInput0, input0Stride, input0Spatial,
                                       &beta,
                                       cudaOutput, k, k * n, batch);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("status = %d\n", (int)status);
        printf("%d %d %d\n", k, n, m);
        printf("Error: cublas error.\n");
        throw("cublas error");
        exit(0);
    }

    xllmCudaFinishInput(input0, cudaInput0);
    xllmCudaFinishInput(input1, cudaInput1);
    xllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool xllmCudaBatchMatMulFP16(const xllm::Data &input0, const xllm::Data &input1, xllm::Data &output,
                            int input0Spatial, int input1Spatial, int outputSpatial,
                            int input0Stride, int input1Stride,
                            int batch, int n, int m, int k, float alpha) {
    float *cudaInput0 = (float *) xllmCudaPrepareInput(input0);
    half *cudaInput1 = (half *) xllmCudaPrepareInput(input1);
    float *cudaOutput = (float *) xllmCudaPrepareOutput(output);
    float beta = 0;
    auto xllmCublasHandle = getxllmCublasHandle();
    cublasStatus_t status;

    half* cudaInput0FP16 = (half*) xllmCudaMalloc(input0.counts * sizeof(half));
    half* cudaOutputFP16 = (half*) xllmCudaMalloc(output.counts * sizeof(half));
    int threadPerBlock = std::min(256, input0.counts);
    xllmCudaFloat2HalfKernel <<< (input0.counts - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaInput0, cudaInput0FP16, input0.counts);

    __half h_alpha = __float2half_rn(alpha), h_beta = __float2half_rn(beta);
    status = cublasHgemmStridedBatched(xllmCublasHandle,
                                       CUBLAS_OP_N, CUBLAS_OP_N,
                                       k, n, m, &h_alpha,
                                       cudaInput1, input1Stride, input1Spatial,
                                       cudaInput0FP16, input0Stride, input0Spatial,
                                       &h_beta,
                                       cudaOutputFP16, k, k * n, batch);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("status = %d\n", (int)status);
        printf("%d %d %d\n", k, n, m);
        printf("Error: cublas error.\n");
        throw("cublas error");
        exit(0);
    }

    threadPerBlock = std::min(256, output.counts);
    xllmCudaHalf2FloatKernel <<< (output.counts - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaOutputFP16, cudaOutput, output.counts);

    xllmCudaFree(cudaInput0FP16);
    xllmCudaFree(cudaOutputFP16);
    xllmCudaFinishInput(input0, cudaInput0);
    xllmCudaFinishInput(input1, cudaInput1);
    xllmCudaFinishOutput(output, cudaOutput);

    return true;
}

__global__ void xllmAddToKernel(float* a, float *b, float alpha, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        a[idx] += b[idx] * alpha;
    }
}

bool xllmCudaAddTo(xllm::Data &input0, const xllm::Data &input1, float alpha) {
    int len = input0.Count(0);
    float *cudaData = (float *) xllmCudaPrepareInput(input0);
    float *input1Data = (float *) xllmCudaPrepareInput(input1);

    int threadPerBlock = std::min(256, len);
    xllmAddToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaData, input1Data, alpha, len);
    xllmCudaFinishInput(input1, input1Data);
    xllmCudaFinishOutput(input0, cudaData);
    return true;
}

__global__ void xllmSiluKernel(float* a, float *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        float x = a[idx];
        b[idx] = x / (1.0 + expf(-x));
    }
}

bool xllmCudaSilu(const xllm::Data &input, xllm::Data &output) {
    int len = input.Count(0);
    float *cudaInput = (float *) xllmCudaPrepareInput(input);
    float *cudaOutput = (float *) xllmCudaPrepareOutput(output);
    int threadPerBlock = std::min(256, len);
    xllmSiluKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaOutput, len);
    xllmCudaFinishInput(input, cudaInput);
    xllmCudaFinishOutput(output, cudaOutput);
    return true;
}


__global__ void xllmMulToKernel(float* a, float *b, float alpha, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        a[idx] *= b[idx] * alpha;
    }
}

bool xllmCudaMulTo(xllm::Data &input0, const xllm::Data &input1, float alpha) {
    int len = input0.Count(0);
    float *cudaData = (float *) xllmCudaPrepareInput(input0);
    float *input1Data = (float *) xllmCudaPrepareInput(input1);

    int threadPerBlock = std::min(256, len);
    xllmMulToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaData, input1Data, alpha, len);
    xllmCudaFinishInput(input1, input1Data);
    xllmCudaFinishOutput(input0, cudaData);
    return true;
}

template <int THREAD_PER_BLOCK>
__global__ void SimpleMask(float* a, float *b, float maskValue, int spatial) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < spatial) {
        if (b[i] > 0.99) {
            a[i] = maskValue;
        }
    }
}

template <int THREAD_PER_BLOCK>
__global__ void xllmAttentionKernel(float *qd, float *kd, float *vd, float *maskd, float *od,
                                       float scale, int q1, int q2, int k1, int v2,
                                       int group, int qstride, int kstride, int vstride, int ostride,
                                       float *qk, float *temp) {
    int o = blockIdx.x;
    qd += o * qstride;
    kd += (o / group) * kstride;
    vd += (o / group) * vstride;
    od += o * ostride;
    qk += o * k1;
    temp += o * k1;
    for (int i = 0; i < q1; i++) {
        for (int j = threadIdx.x; j < k1; j += THREAD_PER_BLOCK) {
            if (maskd && maskd[i * k1 + j] > 0.99) {
                qk[j] = -10000;
                continue;
            }
            float sum = 0.0f;
            float *tempQd = qd + i * q2, *tempKd = kd + j * q2;
            for (int l = 0; l < q2; l++) {
                sum += tempQd[l] * tempKd[l];
            }
            qk[j] = sum * scale;
        }
        __syncthreads();
        xllmSoftmaxKernelInner1Func <THREAD_PER_BLOCK> (qk, temp, k1);
        __syncthreads();
        for (int j = threadIdx.x; j < v2; j += THREAD_PER_BLOCK) {
            float *curInput1 = vd + j;
            float sum = 0.0;
            for (int l = 0; l < k1; l++) {
                sum += temp[l] * curInput1[l * v2];
            }
            od[i * v2 + j] = sum;
        }
        __syncthreads();
    }
}

bool xllmCudaAttention(const xllm::Data &q, const xllm::Data &k, const xllm::Data &v,
                          const xllm::Data &mask, const xllm::Data &output, int group, float scale) {
    int q0 = q.dims[0], q1 = q.dims[1], q2 = q.dims[2], k0 = k.dims[0], k1 = k.dims[1], v2 = v.dims[2];
    float *qd = (float*)q.cudaData;
    float *kd = (float*)k.cudaData;
    float *vd = (float*)v.cudaData;
    float *maskd = mask.dims.size() > 0 ? (float*)mask.cudaData : nullptr;
    float *od = (float*)output.cudaData;
    int batch = (mask.dims.size() == 3) ? mask.dims[0] : 1;
    int maskStride = (mask.dims.size() == 3 ? mask.strides[0] : mask.Count(0));
    if (true) {
        float *qk = (float *) xllmCudaMalloc(q0 * k1 * sizeof(float));
        float *temp = (float *) xllmCudaMalloc(q0 * k1 * sizeof(float));
        xllmAttentionKernel<256> <<<q0, 256>>>(qd, kd, vd, maskd, od,
                                                  scale, q1, q2, k1, v2,
                                                  group, q.strides[0], 
                                                  k.strides[0]*k.expandDims[1]/k.dims[1], 
                                                  v.strides[0]*v.expandDims[1]/v.dims[1], output.strides[0],
                                                  qk, temp);
        xllmCudaFree(qk);
        xllmCudaFree(temp);
        return true;
    }

    if (q1 > 1024) {
        float *qk = (float *) xllmCudaMalloc(q1 * k1 * sizeof(float));
        float beta = 0, one = 1;
        auto fastllmCublasHandle = getxllmCublasHandle();
        cublasStatus_t status;


        for (int i = 0; i < q0; i++) {
            status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                               CUBLAS_OP_T, CUBLAS_OP_N,
                                               k1, q1, q2, &scale,
                                               kd + (i / group) * k.Count(1), k.strides[1], k.Count(1),
                                               qd + i * q.Count(1), q.strides[1], q.Count(1),
                                               &beta,
                                               qk, k1, k1 * q1, 1);
            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("status = %d\n", (int) status);
                printf("Error: cublas error.\n");
                throw ("cublas error");
                exit(0);
            }

            if (maskd) {
                SimpleMask<256> <<< (q1 * k1 / 256) + 1, 256>>>(qk, maskd + (i / (q0 / batch)) * maskStride, -10000, q1 * k1);
            }

            int outer = q1;
            if (k1 < 8) {
                xllmSoftmaxKernelInner1<1> <<< outer, 1 >>>(qk, qk, outer, k1);
            } else if (k1 < 64) {
                xllmSoftmaxKernelInner1<8> <<< outer, 8 >>>(qk, qk, outer, k1);
            } else if (k1 < 512) {
                xllmSoftmaxKernelInner1<64> <<< outer, 64 >>>(qk, qk, outer, k1);
            } else {
                xllmSoftmaxKernelInner1<256> <<< outer, 256 >>>(qk, qk, outer, k1);
            }

            status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                               CUBLAS_OP_N, CUBLAS_OP_N,
                                               v2, q1, k1, &one,
                                               vd + (i / group) * v.Count(1), v.strides[1], v.Count(1),
                                               qk, k1, k1 * q1,
                                               &beta,
                                               od + i * v2 * q1, v2, v2 * q1, 1);
            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("status = %d\n", (int) status);
                printf("Error: cublas error.\n");
                throw ("cublas error");
                exit(0);
            }
        }

        xllmCudaFree(qk);
        // cudaDeviceSynchronize();
        return true;
    }

    if (true) {
        float *qk = (float *) xllmCudaMalloc(q0 * q1 * k1 * sizeof(float));
        float *temp = (float *) xllmCudaMalloc(q0 * q1 * k1 * sizeof(float));
        float beta = 0, one = 1;
        auto fastllmCublasHandle = getxllmCublasHandle();
        cublasStatus_t status;

        status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_T, CUBLAS_OP_N,
                                           k1, q1 * group, q2, &scale,
                                           kd, k.strides[1], k.Count(1)*k.expandDims[1]/k.dims[1],
                                           qd, q.strides[1], q.Count(1) * group,
                                           &beta,
                                           qk, k1, k1 * q1 * group, q0 / group);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("status = %d\n", (int) status);
            printf("Error: cublas error.\n");
            throw ("cublas error");
            exit(0);
        }

        if (maskd) {
            int spatial = q1 * k1, n = 1, m = q0;
            xllmAttentionMaskKernel <256> <<< n * m, 256>>>(qk, maskd, -10000, n, m, spatial);
        }

        int outer = q0 * q1;
        if (k1 < 8) {
            xllmSoftmaxKernelInner1<1> <<< outer, 1 >>>(qk, temp, outer, k1);
        } else if (k1 < 64) {
            xllmSoftmaxKernelInner1<8> <<< outer, 8 >>>(qk, temp, outer, k1);
        } else if (k1 < 512) {
            xllmSoftmaxKernelInner1<64> <<< outer, 64 >>>(qk, temp, outer, k1);
        } else {
            xllmSoftmaxKernelInner1<256> <<< outer, 256 >>>(qk, temp, outer, k1);
        }

        status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_N, CUBLAS_OP_N,
                                           v2, q1 * group, k1, &one,
                                           vd, v.strides[1], v.Count(1)*v.expandDims[1]/v.dims[1],
                                           temp, k1, k1 * q1 * group,
                                           &beta,
                                           od, v2, v2 * q1 * group, q0 / group);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("status = %d\n", (int) status);
            printf("Error: cublas error.\n");
            throw ("cublas error");
            exit(0);
        }
        xllmCudaFree(qk);
        xllmCudaFree(temp);
        // cudaDeviceSynchronize();
        return true;
    }
    return true;
}