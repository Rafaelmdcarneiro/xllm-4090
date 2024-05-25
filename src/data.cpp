#include "data.h"
#include "device.h"

#include "cuda/xllm-cuda.cuh"

namespace xllm{

    void Data::UpdateUnitSize() {
        if (dataType == DataType::FLOAT32) {
            unitSize = 4;
            unitSizeDiv = 1;
        } else if (dataType == DataType::BFLOAT16 || dataType == DataType::FLOAT16) {
            unitSize = 2;
            unitSizeDiv = 1;
        } else if (dataType == DataType::INT8) {
            unitSize = 1;
            unitSizeDiv = 1;
        } else if (dataType == DataType::INT4_NOZERO) {
            unitSize = 1;
            unitSizeDiv = 2;
        }
    }

    Data::Data(DataType type) {
        dataType = type;
        UpdateUnitSize();
    }

    Data::Data(DataType type, const std::vector<int> &dims) : Data::Data(type) {
        counts = 1;
        for (int num : dims) {
            counts *= num;
        }
        bytes = (counts * unitSize - 1) / unitSizeDiv + 1;
        Resize(dims);
    }

    void Data::Resize(const std::vector<int> &dims) {
        this->dims = dims;
        
        strides.clear();
        strides.resize(dims.size(), 1);
        for (int i = dims.size() - 2; i >= 0; i--) {
            strides[i] = dims[i + 1] * strides[i + 1];
        }
    }

    Data::Data(DataType type, const std::vector<int> &dims, const std::vector<float> &data) : Data::Data(type, dims) {
        Allocate();
        // 如果不是float32那么需要量化
        if (type == DataType::FLOAT32) {
            memcpy(cpuData, data.data(), bytes);
        }
    }

    void Data::ToDevice(void *device) {
        BaseDevice *dev = (BaseDevice*)device;
        if (dev->deviceType == "cuda") {
            this->ToDevice(DataDevice::CUDA, dev->deviceIds);
        } else {
            this->ToDevice(DataDevice::CPU, dev->deviceIds);
        }
    }

    void Data::ToDevice(DataDevice device) {
        if (device == DataDevice::CUDA) {
            ToDevice(device, {0});
        } else {
            ToDevice(device, {0});
        }
    }

    void Data::ToDevice(DataDevice device, const std::vector <int> &deviceIds) {
        if (this->dataType == DataType::INT32PARAM) {
            return;
        }

        if (this->dataDevice == device) {
            return;
        }

#ifdef USE_CUDA
        if (this->expandBytes != 0) {
            if (this->dataDevice == DataDevice::CPU) {
                if (device == DataDevice::CUDA) {
                    xllmCudaSetDevice(deviceIds.size() == 0 ? 0 : deviceIds[0]);
                    this->cudaData = xllmCudaMalloc(expandBytes);
                    xllmCudaCopyFromHostToDevice(this->cudaData, this->cpuData, expandBytes);
                    delete[] this->cpuData;
                    this->cpuData = nullptr;
                }
            } else if (this->dataDevice == DataDevice::CUDA) {
                if (device == DataDevice::CPU) {
                    this->cpuData = new uint8_t[expandBytes];
                    xllmCudaCopyFromDeviceToHost(this->cpuData, this->cudaData, expandBytes);
                    xllmCudaFree(this->cudaData);
                    this->cudaData = nullptr;
                } else if (device == DataDevice::CUDA) {
                    xllmCudaSetDevice(this->dataDeviceIds.size() == 0 ? 0 : this->dataDeviceIds[0]);
                    uint8_t *cpuData = new uint8_t[expandBytes];
                    xllmCudaCopyFromDeviceToHost(cpuData, this->cudaData, expandBytes);
                    xllmCudaFree(this->cudaData);

                    xllmCudaSetDevice(deviceIds.size() == 0 ? 0 : deviceIds[0]);
                    this->cudaData = xllmCudaMalloc(expandBytes);

                    xllmCudaCopyFromHostToDevice(this->cudaData, cpuData, expandBytes);
                    delete[] cpuData;
                }
            }
        } else {
             if (this->dataDevice == DataDevice::CPU) {
                xllmCudaSetDevice(deviceIds.size() == 0 ? 0 : deviceIds[0]);
                this->cudaData = xllmCudaMalloc(assignBytes);
                xllmCudaCopyFromHostToDevice(this->cudaData, this->cpuData, assignBytes);
                delete[] this->cpuData;
                this->cpuData = nullptr;
            } else if (this->dataDevice == DataDevice::CUDA) {
                this->cpuData = new uint8_t[assignBytes];
                xllmCudaCopyFromDeviceToHost(this->cpuData, this->cudaData, assignBytes);
                xllmCudaFree(this->cudaData);
                this->cudaData = nullptr;
            }
        }
#endif

        if (deviceIds.size() == 0) {
            this->dataDeviceIds = {0};
        } else {
            this->dataDeviceIds = deviceIds;
        };
        this->dataDevice = device;
    }

    void Data::Allocate() {
        if(assignBytes<bytes){
            FreeSpace();
            MallocSpace(bytes);
            assignBytes = bytes;
        }
    }

    void Data::Allocate(float v) {
        AssertInXLLM(this->dataType == DataType::FLOAT32
                        || this->dataType == DataType::FLOAT16, "Allocate error: Data's type should be float32 or float16.\n");
        this->Allocate();
        if (this->dataType == DataType::FLOAT32) {
            float *f = (float*)cpuData;
            std::fill(f, f + Count(0), v);
        } else if (this->dataType == DataType::FLOAT16) {
            uint16_t *h = (uint16_t*)cpuData;
            std::fill(h, h + Count(0), float_to_half(v));
        }
    }

    void Data::FreeSpace() {
        assignBytes = 0;
        expandBytes = 0;
        if (this->dataDevice == DataDevice::CPU) {
            delete[] this->cpuData;
        } else if (this->dataDevice == DataDevice::CUDA) {
#ifdef USE_CUDA
            xllmCudaFree(this->cudaData);
#else
            ErrorInXLLM("Error: cuda is not supported.\n");
#endif
        }
    }

    void Data::MallocSpace(uint64_t bytes) {
        if (this->dataDevice == DataDevice::CPU) {
            this->cpuData = new uint8_t[bytes];
        } else if (this->dataDevice == DataDevice::CUDA) {
#ifdef USE_CUDA
            this->cudaData = xllmCudaMalloc(bytes);
#else
            ErrorInXLLM("Error: cuda is not supported.\n");
#endif
        }
    }

    Data::~Data() {
        delete[] this->cpuData;
#ifdef USE_CUDA
        if (this->cudaData != nullptr) {
            xllmCudaFree(this->cudaData);
        }
#endif
    }

    Data::Data(const Data &ori) {
        CopyFrom(ori);
    }

    void Data::CopyFrom(const Data &ori) {
        if (ori.dims != this->dims || this->cpuData == nullptr) {
            if (ori.dims.size() == 0) {
                delete[] this->cpuData;
                this->dataType = ori.dataType;
                this->UpdateUnitSize();
                this->dims.resize(0);
                this->cpuData = nullptr;
                return;
            }
            this->dataType = ori.dataType;
            this->Resize(ori.dims);
            counts = 1;
            for (int num : dims) {
                counts *= num;
            }
            this->bytes = (counts * unitSize - 1) / unitSizeDiv + 1;
            this->Allocate();
        }
        std::memcpy(this->cpuData, ori.cpuData, bytes);
    }

    void Data::Reshape(const std::vector<int> &dims) {
        int negative_index = -1;
        uint64_t new_counts = 1;
        for (int i = 0; i < dims.size(); i++) {
            if (dims[i] < 0) {
                if (negative_index == -1) {
                    negative_index = i;
                } else {
                    // dims只能包含一个负数索引
                    ErrorInXLLM("Reshape error.\n");
                }
            } else {
                new_counts *= dims[i];
            }
        }
        std::vector <int> outputDims = dims;
        if (negative_index == -1) {
            AssertInXLLM(new_counts == counts, "Reshape error.\n");
        } else {
            AssertInXLLM(new_counts != 0, "Reshape error.\n");
            AssertInXLLM(counts % new_counts == 0, "Reshape error.\n");
            outputDims[negative_index] = counts / new_counts;
        }
        Resize(outputDims);
    }

    uint64_t Data::Count(int i)  const{
        if (i >= this->dims.size()) {
            return 1;
        }
        if (i - 1 >= 0 && i - 1 < this->strides.size()) {
            return this->strides[i - 1];
        }
        return this->dims[i] * this->strides[i];
    }

    void Data::Expansion(const std::vector<int> &dims) {
        
        expandDims = dims;
        expandCounts = 1;
        for (int dim : dims) {
            expandCounts *= dim;
        }
        expandBytes = (expandCounts * unitSize - 1) / unitSizeDiv + 1;

        // 如果还没有分配空间
        if (this->dims.size() == 0 || assignBytes == 0) {
            this->MallocSpace(expandBytes);
            assignBytes = expandBytes;
            return;
        }

        AssertInXLLM(dims.size() == this->dims.size(), "Expansion error: real dims's size should equal to expansion dims's size.\n");
        for (int i = 0; i < dims.size(); i++) {
            AssertInXLLM(dims[i] == -1 || dims[i] >= this->dims[i], "Expansion error: real size should <= expansion size.\n");
        }

        // 要扩张哪一个维度
        int axis = -1;
        for (int i = 0; i < dims.size(); i++) {
            if (this->dims[i] < dims[i]) {
                axis = i;
                break;
            }
        }

        // 把原来的数据拷贝到新的空间
        if (this->dataDevice == DataDevice::CPU) {
            uint8_t *old = this->cpuData;
            MallocSpace(expandBytes);
            int inputStride = this->Count(axis);
            int outer = this->counts / inputStride;
            for (int o = 0; o < outer; o++) {
                memcpy(this->cpuData + o * inputStride/this->dims[axis]*dims[axis] * unitSize,
                        old + o * inputStride * unitSize,
                        inputStride * unitSize);
            }
            delete[] old;
        } else if (this->dataDevice == DataDevice::CUDA) {
#ifdef USE_CUDA
                uint8_t *old = (uint8_t*)this->cudaData;
                MallocSpace(expandBytes);
                int inputStride = this->Count(axis);
                int outer = this->counts / inputStride;
                xllmCudaMemcpy2DDeviceToDevice((uint8_t*)this->cudaData, inputStride/this->dims[axis]*dims[axis] * unitSize,
                                            (uint8_t*)old, inputStride * unitSize, inputStride * unitSize, outer);
                xllmCudaFree(old);
                xllmCudaClearBigBuffer();
#else
                ErrorInXLLM("Error: cuda is not supported.\n");
#endif
            }
    }

    void Data::CalcWeightSum() {
        if (this->weightSum.size() > 0) {
            return;
        }
        int n = this->dims[0], m = this->dims[1];
        if (this->dataType == DataType::INT8) {
            weightSum.resize(n);
            for (int i = 0; i < n; i++) {
                int j = 0;
#ifdef __AVX__
                __m256i acc = _mm256_setzero_si256();
                const __m256i ones = _mm256_set1_epi16(1);
                for (; j + 31 < m; j += 32) {
                    __m256i ax = _mm256_loadu_si256((const __m256i *) (cpuData + i * m + j));
                    __m256i mx0 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(ax, 0));
                    __m256i mx1 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(ax, 1));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(mx0, ones));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(mx1, ones));
                }
                weightSum[i] += I32sum(acc);
#endif
                for (; j < m; j++) {
                    weightSum[i] += cpuData[i * m + j];
                }
            }
        } else if (this->dataType == DataType::INT4 || this->dataType == DataType::INT4_NOZERO) {
            weightSum.resize(n);
            for (int i = 0; i < n; i++) {
                int j = 0;
#ifdef __AVX__
	            __m256i acc = _mm256_setzero_si256();
	            const __m256i lowMask = _mm256_set1_epi8(0xf);
	            const __m256i ones = _mm256_set1_epi16(1);
	            for (; j + 31 < m; j += 32) {
		            __m128i orix = _mm_loadu_si128((const __m128i *) (cpuData + (i * m + j) / 2));
		            __m256i bytex = _mm256_set_m128i(_mm_srli_epi16(orix, 4), orix);
		            __m256i bx = _mm256_and_si256(lowMask, bytex);

		            __m256i mx0 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(bx, 0));
		            __m256i mx1 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(bx, 1));

		            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(mx0, ones));
		            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(mx1, ones));
	            }
	            weightSum[i] += I32sum(acc);
#endif
                for (; j + 1 < m; j += 2) {
	                int id = (i * m + j) / 2;
	                weightSum[i] += (cpuData[id] & 0xF) + (cpuData[id] >> 4);
                }
                for (; j < m; j++) {
                    int id = (i * m + j) / 2;
                    if ((i * m + j) % 2) {
                        weightSum[i] += (cpuData[id] & 0xF);
                    } else {
                        weightSum[i] += (cpuData[id] >> 4);
                    }
                }
            }
        }
    }

    void Data::removeBatch(std::vector<int> removedBatch, int batch){
        if (dataDevice == DataDevice::CPU){
            // for (int i = d; i < batch-1; i++)
            //     memcpy(cpuData + i*bytes/batch, cpuData + (i+1)*bytes/batch, bytes/batch);
            // bytes -= bytes/batch;
            // counts -= strides[0];
            // dims[0]--;
        } else {
            // void *old = this->cudaData;
            int removedBatchNum = removedBatch.size();
            float shrink = (float)(batch-removedBatchNum)/batch;
            expandBytes *= shrink;
            int strideBytes = expandDims[1] * expandDims[2] * dims[0]/batch * unitSize;
            std::vector<int> remainBatch;
            std::vector<int> batchVetor;
            for (int i = 0; i < batch; ++i) {
                batchVetor.push_back(i);
            }
            for (int x : batchVetor) {
                if (std::find(removedBatch.begin(), removedBatch.end(), x) == removedBatch.end()) {
                    remainBatch.push_back(x);
                }
            }
#ifdef USE_CUDA
            // this->cudaData = xllmCudaMalloc(expandBytes);
            xllmCudaCopyFromDeviceToDevice(remainBatch.size(), strideBytes,
                cudaData, cudaData, remainBatch);
            // xllmCudaFree(old);
            // xllmCudaClearBigBuffer();
#endif            
            dims[0] *= shrink;
            counts *= shrink;
            expandDims[0] *= shrink;
            expandCounts *= shrink;
            assignBytes *= shrink;
        }
    }    
}
