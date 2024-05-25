#include <cfloat>

#include "operator.h"
#include "data.h"

namespace xllm{
    // uint16_t -> float32
    struct FP16ToFP32Manager {
        float dict[65536];  // 2^16

        FP16ToFP32Manager() {
            for (uint16_t i = 0; i < 65535; i++) {
                dict[i] = half_to_float(i);
            }
        }
    } fp16tofp32;

    void Embedding(const Data &input, Data &weight, Data &output) {
        output.Allocate();
        int embSize = weight.dims[1];
        float *inputData = (float*)input.cpuData;
        if (weight.dataType == DataType::FLOAT32) {
            float *outputData = (float *) output.cpuData;
            float *weightData = (float *) weight.cpuData;
            for (int i = 0; i < input.counts; i++) {
                int token = (int) (inputData[i] + 1e-9);
                memcpy(outputData + i * embSize, weightData + token * embSize, embSize * sizeof(float));
            }
        } else if (weight.dataType == DataType::BFLOAT16){
            uint16_t *outputData = (uint16_t *) output.cpuData;
            uint16_t *weightData = (uint16_t *) weight.cpuData;
            for (int i = 0; i < input.counts; i++) {
                int token = (int) (inputData[i] + 1e-9);
                for (int j = 0; j < embSize; j++) {
                    outputData[i * embSize * 2 + j * 2] = 0;    // 低位是0
                    outputData[i * embSize * 2 + j * 2 + 1] = weightData[token * embSize + j];  // 高位是uint16_t
                }
            }
        }
    }

    void RMSNorm(const Data &input, Data &weight, Data &output, float eps) {
        output.Allocate();
        int inner = input.dims.back();
        int outer = input.counts / inner;

        if (output.dataType == DataType::FLOAT32) {
            float *inputData = (float *) input.cpuData;
            float *outputData = (float *) output.cpuData;
            float *weightData = (float *) weight.cpuData;

            for (int i = 0; i < outer; i++) {
                float mean = 0.f;
                int j = 0;
                for (; j < inner; j++) {
                    mean += inputData[j] * inputData[j];
                }
                float scale = 1.0 / sqrt(mean / inner + eps);
                j = 0;
                for (; j < inner; j++) {
                    outputData[j] = inputData[j] * scale * weightData[j];
                }

                inputData += inner;
                outputData += inner;
            }
        } else if (output.dataType == DataType::FLOAT16) {
            float *inputData = (float*) input.cpuData;
            uint16_t *outputData = (uint16_t *) output.cpuData;
            float *weightData = (float *) weight.cpuData;

            for (int i = 0; i < outer; i++) {
                float mean = 0.f;
                int j = 0;
                for (; j < inner; j++) {
                    float x = inputData[j];
                    mean += x * x;
                }
                float scale = 1.0 / sqrt(mean / inner + eps);
                j = 0;
                for (; j < inner; j++) {
                    outputData[j] = float_to_half(inputData[j] * scale * weightData[j]);
                }

                inputData += inner;
                outputData += inner;
            }
        } 
        else {
            ErrorInXLLM("RMSNorm error: unsupport dataType.\n");
        }
    }

    // inputData(n, m) * weightData(m, end-st) = outputData(n, end-st)
    void FloatLinearPart(float *inputData, float *weightData, float *biasData, float *outputData,
                            int n, int m, int k, int st, int end) {
            for (int i = 0; i < n; i++) {
                for (int j = st; j < end; j++) {
                    float now = biasData ? biasData[j] : 0.0f;
                    int l = 0;
    #ifdef __AVX2__
                    __m256 vsum = _mm256_setzero_ps();
                    for (; l + 7 < m; l += 8) {
                        __m256 vi = _mm256_loadu_ps(inputData + i * m + l);
                        __m256 vw = _mm256_loadu_ps(weightData + j * m + l);
                        vsum = _mm256_fmadd_ps(vi, vw, vsum);
                    }
                    now += Floatsum(vsum);
    #endif
                    for (; l < m; l++) {
                        now += inputData[i * m + l] * weightData[j * m + l];
                    }
                    outputData[i * k + j] = now;
                }
            }
        }

    // inputData(float32) * weightData(float16) = outputData(float32)
    void Float16LinearPart(float *inputData, uint16_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
        // 遍历行
        for (int i = 0; i < n; i++) {
            // 遍历列
            for (int j = st; j < end; j++) {
                float now = biasData ? biasData[j] : 0.0f;
                int l = 0;
#ifdef __AVX2__
                __m256 vsum = _mm256_setzero_ps();
                for (; l + 7 < m; l += 8) {
                    // 256位单精度浮点向量
                    __m256 vi = _mm256_loadu_ps(inputData + i * m + l);
                    // _mm_loadu_si128: load 128位的整数
                    // _mm256_cvtph_ps: float16 -> float32 
                    __m256 vw = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *) (weightData + j * m + l)));
                    vsum = _mm256_fmadd_ps(vi, vw, vsum);
                }
                now += Floatsum(vsum);
#endif
                for (; l < m; l++) {
                    now += inputData[i * m + l] * fp16tofp32.dict[weightData[j * m + l]];
                }
                outputData[i * k + j] = now;
            }
        }
    }

    // inputData(float16) * weightData(float16) = outputData(float16)
    void Float16xFloat16LinearPart(uint16_t *inputData, uint16_t *weightData, float *biasData, float *outputData,
                           int n, int m, int k, int st, int end) {
        for (int i = 0; i < n; i++) {
            for (int j = st; j < end; j++) {
                float now = biasData ? biasData[j] : 0.0f;
                int l = 0;
                for (; l < m; l++) {
                    now += inputData[i * m + l] * weightData[j * m + l];
                }
                outputData[i * k + j] = now;
            }
        }
    }

#ifdef __AVX__
#ifdef __AVX2__
    // TODO: 计算两个数组的点积 
    int DotU8U8(uint8_t *a, uint8_t *b, int n) {
        // 全零的256位整数向量，用于累加乘积的结果
        __m256i acc = _mm256_setzero_si256();
        int i = 0;
        int ans = 0;
        // const __m256i lowMask = _mm256_set1_epi8(0xf);
        // 全1的16位整数向量，用于将8位整数扩展为16位整数
        const __m256i ones = _mm256_set1_epi16(1);
        // 全1的8位整数向量，用于将8位整数扩展为32位整数
        const __m256i ones8 = _mm256_set1_epi8(1);
        // 全-128(10000000)的8位整数向量
        const __m256i xors = _mm256_set1_epi8(-128);
        for (; i + 31 < n; i += 32) {
            // 加载256位整数向量
            __m256i bx = _mm256_loadu_si256((const __m256i *) (a + i));
            __m256i by = _mm256_loadu_si256((const __m256i *) (b + i));

            // 将by第一位进行翻转
            by = _mm256_xor_si256(by, xors);
            by = _mm256_add_epi8(by, _mm256_and_si256(_mm256_cmpeq_epi8(by, xors), ones8));

            by = _mm256_sign_epi8(by, bx);
            bx = _mm256_sign_epi8(bx, bx);

            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(bx, by), ones));
        }
        for (; i < n; i++) {
            ans += ((int8_t*)a)[i] * ((int)b[i] - 128);
        }

        return ans + I32sum(acc);
    };
#else
    int DotU8U8(uint8_t *a, uint8_t *b, int n) {
        __m256i acc = _mm256_setzero_si256();

        int i = 0;
        int ans = 0;
        for (; i + 31 < n; i += 32) {
            __m256i bx = _mm256_loadu_si256((const __m256i *) (a + i));
            __m256i by = _mm256_loadu_si256((const __m256i *) (b + i));

            __m256i mx0 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(bx, 0));
            __m256i mx1 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(bx, 1));

            __m256i my0 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(by, 0));
            __m256i my1 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(by, 1));

            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(mx0, my0));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(mx1, my1));
        }
        for (; i < n; i++) {
            ans += a[i] * b[i];
        }

        return ans + I32sum(acc);
    };
#endif
    int DotU4U8(uint8_t *a, uint8_t *b, int n) {
        __m256i acc = _mm256_setzero_si256();

        int i = 0;
        int ans = 0;
        const __m256i lowMask = _mm256_set1_epi8(0xf);
        const __m256i ones = _mm256_set1_epi16(1);
        for (; i + 31 < n; i += 32) {
            __m128i orix = _mm_loadu_si128((const __m128i *) (a + i / 2));
            __m256i bytex = _mm256_set_m128i(_mm_srli_epi16(orix, 4), orix);
            __m256i bx = _mm256_and_si256(lowMask, bytex);
            __m256i by = _mm256_loadu_si256((const __m256i *) (b + i));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(by, bx), ones));
        }
        for (; i < n; i++) {
            ans += a[i] * b[i];
        }

        return ans + I32sum(acc);
    };
#endif

    //a = [n, m], b = [k, m], c = aT(b') = [n, k]
    void Multiply(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k, int kstride) {
#if defined(__AVX__)
        int block = 0;
        for (; block < n; block++) {
            uint8_t *weightWalk = b;
            uint8_t *inputStart = a + block * m;

            for (int i = 0; i < k; i++) {
                uint8_t *inputWalk = inputStart;

                c[block * kstride + i] = DotU8U8(inputWalk, weightWalk, m);
                weightWalk += m;
            }
        }
#else
        int block = 0;
	    for (; block < n; block++) {
		    uint8_t *weightWalk = b;
		    uint8_t *inputStart = a + block * m;

		    for (int i = 0; i < k; i++) {
			    int value = 0;
			    uint8_t *inputWalk = inputStart;
			    for (int j = 0; j < m; j++) {
				    value += (int)(*(weightWalk++)) * (*(inputWalk++));
			    }

			    c[block * kstride + i] = value;
		    }
	    }
#endif
    }

    //a = [n, m], b = [k, m], c = aT(b') = [n, k]
    void MultiplyMultiThread(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k, int threadNum) {
        int per = k / threadNum;
        int cur = 0;
        if (threadNum == 1) {
            Multiply(a, b + cur * m, c + cur, n, m, k - cur, k);
        } else {
            auto pool = GetPool();
            std::vector<std::future<void> > futures;
            for (int i = 0; i < threadNum; i++) {
                int end = cur + per + (cur + per * (threadNum - i) < k);
                if (i == threadNum - 1) {
                    end = k;
                }
                futures.push_back(pool->enqueue(Multiply, a, b + cur * m, c + cur, n, m, end - cur, k));
                cur = end;
            }
            for (int i = 0; i < futures.size(); i++) {
                futures[i].get();
            }
        }
    }

    // 多维矩阵（可以转换为二维矩阵）*二维矩阵乘法：input(n,m) * weight(k,m) + bias = output(n,k)
    void Linear(const Data &input, Data &weight, Data &output) {
        output.Allocate(0);
        // auto st = std::chrono::system_clock::now();
        Data bias;

        int n = input.counts / input.dims.back();  // 前面的维度打包到一个维度
        int m = input.dims.back();
        int k = output.dims.back();
        float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;

        if (input.dataType == DataType::FLOAT32 && output.dataType == DataType::FLOAT32) {
            if (weight.dataType == DataType::FLOAT32) {
                float *inputData = (float *) input.cpuData;
                float *weightData = (float *) weight.cpuData;
                float *outputData = (float *) output.cpuData;

                int threadNum = GetThreads();
                int per = k / threadNum;    // 一个线程负责per列的计算
                int cur = 0;
                auto pool = GetPool();
                std::vector<std::future<void> > futures;
                for (int i = 0; i < threadNum - 1; i++) {
                    int end = cur + per + (cur + per * (threadNum - i) < k);
                    futures.push_back(pool->enqueue(FloatLinearPart, inputData, weightData, biasData, outputData,
                                                   n, m, k, cur, end));
                    cur = end;
                }

                FloatLinearPart(inputData, weightData, biasData, outputData, n, m, k, cur, k);  // 如果k不能被threadNum整除
                for (int i = 0; i < futures.size(); i++) {
                    futures[i].get();
                }
            } else if (weight.dataType == DataType::FLOAT16) {
                float *inputData = (float *) input.cpuData;
                // 大部分CPU不支持FP16类型，但是支持uint16_t
                uint16_t *weightData = (uint16_t *) weight.cpuData;
                float *outputData = (float *) output.cpuData;
                int threadNum = GetThreads();
                int per = k / threadNum;
                int cur = 0;
                auto pool = GetPool();
                std::vector<std::future<void> > futures;
                for (int i = 0; i < threadNum - 1; i++) {
                    int end = cur + per + (cur + per * (threadNum - i) < k);
                    futures.push_back(pool->enqueue(Float16LinearPart, inputData, weightData, biasData, outputData,
                                                   n, m, k, cur, end));
                    cur = end;
                }

                Float16LinearPart(inputData, weightData, biasData, outputData, n, m, k, cur, k);
                for (int i = 0; i < futures.size(); i++) {
                    futures[i].get();
                }
            } else if (weight.dataType == DataType::INT8) {
                float *inputData = (float *) input.cpuData;
                uint8_t *weightData = (uint8_t *) weight.cpuData;
                float *outputData = (float *) output.cpuData;
                weight.CalcWeightSum();

                // 按行量化input: fp32->int8
                std::vector<LowBitConfig> inputConfigs;
                for (int i = 0; i < n; i++) {
                    float minValue = 1e9, maxValue = -1e9;
                    for (int j = 0; j < m; j++) {
                        minValue = std::min(minValue, inputData[i * m + j]);
                        maxValue = std::max(maxValue, inputData[i * m + j]);
                    }
                    inputConfigs.push_back(LowBitConfig(minValue, maxValue, 8, 0));
                }
                std::vector<uint8_t> uinput;
                uinput.resize(n * m);
                for (int i = 0; i < n * m; i++) {
#ifdef __AVX2__
                    uinput[i] = inputConfigs[i / m].quantization(inputData[i]);
                    uinput[i] = (uinput[i] + !uinput[i]) ^ 128;
#else
                    uinput[i] = inputConfigs[i / m].quantization(inputData[i]);
#endif
                }

                MultiplyMultiThread(uinput.data(), weightData, (int32_t *) outputData, n, m, k, GetThreads());
                // 遍历输出矩阵的行
                for (int i = 0; i < n; i++) {
                    uint32_t inputSum = 0;  // 每一行的和
                    for (int j = 0; j < m; j++) {
#ifdef __AVX2__
                        inputSum += uinput[i * m + j] ^ 128;
#else
                        inputSum += uinput[i * m + j];
#endif
                    }

                    // 遍历输出矩阵的列
                    for (int j = 0; j < k; j++) {
                        int value = ((int32_t *) outputData)[i * k + j];
#ifdef __AVX2__
                        value += (128 * weight.weightSum[j]);
                        value += (128 * inputSum);
                        value -= m * 128 * 128;
#endif
                        value -= weight.weightSum[j] * inputConfigs[i].zeroPoint;
                        value -= inputSum * weight.perChannelsConfigs[j].zeroPoint;
                        value += (int) inputConfigs[i].zeroPoint * weight.perChannelsConfigs[j].zeroPoint * m;
                        outputData[i * k + j] = weight.perChannelsConfigs[j].scale * inputConfigs[i].scale * value +
                                                (biasData == nullptr ? 0.0 : biasData[j]);
                    }
                }
            } else {
                ErrorInXLLM("Linear error: unsupport weight's dataType.\n");
            }
        } else if (input.dataType == DataType::FLOAT16 && weight.dataType == DataType::FLOAT16) {
                uint16_t *inputData = (uint16_t *) input.cpuData;
                uint16_t *weightData = (uint16_t *) weight.cpuData;
                float *outputData = (float *) output.cpuData;
                float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;
                int threadNum = GetThreads();
                int per = k / threadNum;
                int cur = 0;
                auto pool = GetPool();
                std::vector<std::future<void> > futures;
                for (int i = 0; i < threadNum - 1; i++) {
                    int end = cur + per + (cur + per * (threadNum - i) < k);
                    futures.push_back(pool->enqueue(Float16xFloat16LinearPart, inputData, weightData, biasData, outputData,
                                                   n, m, k, cur, end));
                    cur = end;
                }

                Float16xFloat16LinearPart(inputData, weightData, biasData, outputData, n, m, k, cur, k);
                for (int i = 0; i < futures.size(); i++) {
                    futures[i].get();
                }
        } else {
            ErrorInXLLM("Linear error: unsupport input's dataType.\n");
        }

        // float spend = GetSpan(st, std::chrono::system_clock::now());
        // float gops = (float)2* n * m * k / spend / 1e9;
        // printf("n = %d, m = %d, k = %d, spend %f s, gops = %f\n", n, m, k, spend, gops);
    }

    void LlamaRotatePosition2D(Data &input, const Data &positionIds, Data &sinData, Data &cosData, int rotaryDim) {

        int bsz = input.dims[0], seqlen = input.dims[1];
        int spatial = input.Count(2);
        int n = input.dims[2], m = input.dims[3];
        int stride = (int)sinData.dims[1];
        for (int b = 0; b < bsz; b++) {
            for (int l = 0; l < seqlen; l++) {
                int index = (int) ((float *) positionIds.cpuData)[b * positionIds.dims.back() + l];
                float *sin = ((float *) sinData.cpuData) + stride * index;
                float *cos = ((float *) cosData.cpuData) + stride * index;
                float *d = (float *) input.cpuData + (b * seqlen + l) * spatial;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < rotaryDim && j < m / 2; j++) {
                        float a = d[j], b = d[j + m / 2];
                        d[j] = a * cos[j] - b * sin[j];
                        d[j + m / 2] = a * sin[j] + b * cos[j];
                    }

                    d += m;
                }
            }
        }
    }

    void Transpose4x4(float *pDst, float *pSrc, int dstStride, int srcStride, int n, int m) {
        if (n < 4 || m < 4) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    pDst[j * dstStride + i] = pSrc[i * srcStride + j];
                }
            }

            return;
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                pDst[j * dstStride + i] = pSrc[i * srcStride + j];
            }
        }
    }

    void Transpose(float *pDst, float *pSrc, int dstStride, int srcStride, int n, int m) {
        int per = 4;
        for (int i = 0; i < n; i += per) {
            for (int j = 0; j < m; j += per) {
                Transpose4x4(pDst + j * dstStride + i,
                             pSrc + i * srcStride + j,
                             dstStride, srcStride,
                             std::min(per, n - i),
                             std::min(per, m - j));
            }
        }
    }

    void Permute(Data &input, Data &output, std::vector <int> axis) {

        output.Allocate();
        uint8_t *tmpData = (uint8_t *) output.cpuData;
        uint8_t *curData = (uint8_t *) input.cpuData;

        if (axis == std::vector <int> {1, 2, 0} && input.dataType == DataType::FLOAT32) {
            int n = input.dims[0];
            int m = input.Count(1);

            int threadNum = 1;
            int per = m / threadNum;
            int cur = 0;
            auto pool = GetPool();
            std::vector <std::future <void> > futures;
            for (int i = 0; i < threadNum - 1; i++) {
                int end = cur + per + (cur + per * (threadNum - i) < m);
                futures.push_back(pool->enqueue(Transpose, ((float*)tmpData) + cur * n, ((float*)curData) + cur, n, m, n, end - cur));
                cur = end;
            }
            Transpose(((float*)tmpData) + cur * n, ((float*)curData) + cur, n, m, n, m - cur);
            for (int i = 0; i < futures.size(); i++) {
                futures[i].get();
            }
        } else if (axis == std::vector <int> {1, 0, 2}) {
            int n = input.dims[0];
            int m = input.dims[1];
            int k = input.dims[2];
            int unitSize = input.unitSize;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    memcpy(tmpData + (j * n + i) * k * unitSize, curData + (i * m + j) * k * unitSize, k * unitSize);
                }
            }
        } else if (axis == std::vector <int> {2, 0, 1, 3}) {
            int n = input.dims[0] * input.dims[1];
            int m = input.dims[2];
            int k = input.dims[3];
            int unitSize = input.unitSize;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    memcpy(tmpData + (j * n + i) * k * unitSize, curData + (i * m + j) * k * unitSize, k * unitSize);
                }
            }
        } else if (axis == std::vector<int> {0, 2, 1, 3}) {
            int b = input.dims[0];
            int n = input.dims[1];
            int m = input.dims[2];
            int k = input.dims[3];
            int unitSize = input.unitSize;
            for (int o = 0; o < b; o++) {
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < m; j++) {
                        memcpy(tmpData + (j * n + i) * k * unitSize, curData + (i * m + j) * k * unitSize, k * unitSize);
                    }
                }
                tmpData += output.Count(1) * unitSize;
                curData += input.Count(1) * unitSize;
            }
        } else {
            std::vector<int> oldSteps;
            std::vector<int> newSteps;
            int count = input.Count(0);
            auto oldPos = new int[count];
            for (int i = 0; i < axis.size(); i++) {
                oldSteps.push_back(input.Count(i + 1));
                newSteps.push_back(output.Count(i + 1));
            }

            for (int i = 0; i < count; ++i) {
                int old = 0;
                int idx = i;
                for (int j = 0; j < axis.size(); ++j) {
                    int order = axis[j];
                    old += (idx / newSteps[j]) * oldSteps[order];
                    idx %= newSteps[j];
                }
                oldPos[i] = old;
            }

            if (input.unitSize == 4) {
                for (int i = 0; i < count; ++i) {
                    ((float*)tmpData)[i] = ((float*)curData)[oldPos[i]];
                }
            } else if (input.unitSize == 2) {
                for (int i = 0; i < count; ++i) {
                    ((uint16_t*)tmpData)[i] = ((uint16_t*)curData)[oldPos[i]];
                }
            } else if (input.unitSize == 1) {
                for (int i = 0; i < count; ++i) {
                    ((uint8_t*)tmpData)[i] = ((uint8_t*)curData)[oldPos[i]];
                }
            }

            delete[] oldPos;
        }
    }

    void PermuteSelf(Data &input, std::vector <int> axis) {

        AssertInXLLM(input.dataType == DataType::FLOAT32 ||
                        input.dataType == DataType::FLOAT16, "Permute error: datatype should be float32 or float16.");
        AssertInXLLM(axis.size() == input.dims.size(), "Permute error: axis's size should be equal to data's shape's size.");

        std::vector<int> new_dims;
        for (int i = 0; i < axis.size(); i++) {
            new_dims.push_back(input.dims[axis[i]]);
        }
        
        bool same = false;
        // 下面几种情况不需要移动数据
        // 例如: (1,2,3) -> (2,3,1) / (2,1,3)
        same |= ((axis == std::vector <int>{1, 2, 0} || axis == std::vector <int>{1, 0, 2}) && (input.dims[0] == 1 || input.dims[1] == 1));
        same |= ((axis == std::vector <int>{2, 0, 1, 3}) && input.dims[2] == 1);
        same |= ((axis == std::vector <int>{0, 2, 1, 3}) && (input.dims[1] == 1 || input.dims[2] == 1));
        if (same) {
            input.Resize(new_dims);
            return;
        }

        auto tmp = new Data(DataType::FLOAT32, new_dims);
        Permute(input, *tmp, axis);

        memcpy(input.cpuData, tmp->cpuData, input.unitSize * input.counts);
        input.Resize(tmp->dims);
        delete tmp;
    }

    void CatDirect(Data &input0, Data &input1, int axis) {

        AssertInXLLM((input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32) ||
                        (input0.dataType == DataType::FLOAT16 && input1.dataType == DataType::FLOAT16),
                        "CatDirect's input's type should be float32.\n");

        // input0还没有数据
        if (input0.dims.size() == 0) {
            input0.counts = input1.counts;
            input0.bytes = input1.bytes;
            input0.Resize(input1.dims);

            AssertInXLLM(input0.expandDims.size() == input1.dims.size() &&
                            input1.dims[axis] <= input0.expandDims[axis],
                            "CatDirect Error: input0's expansion size is not enough.\n");

            int outer = input1.counts / input1.Count(axis);
            int input1Stride = input1.Count(axis);
            int input0Stride = input1Stride/input1.dims[axis] * input0.expandDims[axis];
            int unitSize = input0.unitSize;
            for (int o = 0; o < outer; o++) {
                memcpy(input0.cpuData + o * input0Stride * unitSize,
                       input1.cpuData + o * input1Stride * unitSize,
                       input1Stride * unitSize);
            }

            return;
        }

        // input0已有数据
        AssertInXLLM(input0.dims.size() == input1.dims.size(), "Cat Error: input's shape's size should be same.\n");
        int dimsLen = input0.dims.size();
        // axis = (axis % dimsLen + dimsLen) % dimsLen;

        for (int i = 0; i < dimsLen; i++) {
            if (i != axis) {
                AssertInXLLM(input0.dims[i] == input1.dims[i], "Cat Error: input's shape doesn't match.");
            }
        }

        std::vector <int> dims = input0.dims;
        std::vector <int> oldDims = dims;
        dims[axis] += input1.dims[axis];
        input0.Resize(dims);
        input0.counts += input1.counts;
        int unitSize = input0.unitSize;
        input0.bytes += (input1.Count(axis) * unitSize - 1) / input0.unitSizeDiv + 1;

        int outer = input0.counts / input0.Count(axis);
        int input1Stride = input1.Count(axis);
        int input0Stride = input1Stride/input1.dims[axis] * input0.expandDims[axis];
        int inner = input0.strides[axis];
        for (int o = 0; o < outer; o++) {
            memcpy(input0.cpuData + o * input0Stride * unitSize + oldDims[axis] * inner * unitSize,
                   input1.cpuData + o * input1Stride * unitSize,
                   input1Stride * unitSize);
        }
    }

    void MatMulTransBSingle(float *input0Base, float *input1Base, float *outputBase,
                                int input0Spatial, int input1Spatial, int outputSpatial,
                                int n, int m, int k, float alpha, int st, int end) {
        for (int b = st; b < end; b++) {
                float *input0Data = input0Base + b * input0Spatial;
                float *input1Data = input1Base + b * input1Spatial;
                float *outputData = outputBase + b * outputSpatial;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < k; j++) {
                        float now = 0.0f;
                        int l = 0;
    #if defined(__AVX__)
                        __m256 vsum = _mm256_set1_ps(0.0f);
                        for (; l + 7 < m; l += 8) {
                            __m256 vx = _mm256_loadu_ps((const float *) (input0Data + i * m + l));
                            __m256 vy = _mm256_loadu_ps((const float *) (input1Data + j * m + l));
                            vsum = _mm256_add_ps(vsum, _mm256_mul_ps(vx, vy));
                        }
                        now += Floatsum(vsum);
    #endif
                        for (; l < m; l++) {
                            now += input0Data[i * m + l] * input1Data[j * m + l];
                        }
                        outputData[i * k + j] = now * alpha;
                    }
                }
            }
        }

    // 三维矩阵乘法: input0(batch, n, m) * input1(batch, k, m)^T = output(batch, n, k)
    // input1内存不连续，不同batch之间的矩阵乘法要分开计算
    void MatMulTransB(Data &input0, Data &input1, Data &output, float alpha) {
        output.Allocate();

        // 一个batch有多少个数
        int input0Spatial = input0.Count(input0.dims.size() - 2);   // n*m
        int input1Spatial = input1.dims.back()*input1.expandDims[input1.dims.size() - 2];   // m*expandDims[1]
        int outputSpatial = output.Count(output.dims.size() - 2);    // n*k

        // int input0Stride = input0.strides[input0.dims.size() - 2];     // m
        // int input1Stride = input1.strides[input1.dims.size() - 2];     // m

        int batch = input0.counts / input0Spatial;
        int n = input0.dims[input0.dims.size() - 2];
        int m = input0.dims.back();
        int k = input1.dims[input1.dims.size() - 2];

        int threadNum = GetThreads();
        if (batch * n * m * k < 64 * 4096) {
            threadNum = 1;
        }
        // threadNum = std::min(threadNum, 4);
        int per = batch / threadNum;  // 一个线程负责per个batch的计算
        int cur = 0;
        auto pool = GetPool();
        std::vector <std::future <void> > futures;
        if (input0.dataType == DataType::FLOAT32) {
            for (int i = 0; i < threadNum - 1; i++) {
                int end = cur + per + (cur + per * (threadNum - i) < batch);
                futures.push_back(pool->enqueue(MatMulTransBSingle,
                                               (float *) input0.cpuData, (float *) input1.cpuData,
                                               (float *) output.cpuData,
                                               input0Spatial, input1Spatial, outputSpatial,
                                               n, m, k, alpha, cur, end));
                cur = end;
            }
            MatMulTransBSingle((float *) input0.cpuData, (float *) input1.cpuData, (float *) output.cpuData,
                               input0Spatial, input1Spatial, outputSpatial,n, m, k, alpha, cur, batch);
        } 
        for (int i = 0; i < futures.size(); i++) {
            futures[i].get();
        }
    }


    void AttentionMask(Data &input, const Data &mask, float maskValue) {
        int spatial = input.Count(2), n = input.dims[0], m = input.dims[1];

        AssertInXLLM(mask.dataType == DataType::FLOAT32, "AttentionMask: mask's datatype should be float32.");
        if (input.dataType == DataType::FLOAT32) {
            float *maskData = (float *) mask.cpuData;
            float *attnData = (float *) input.cpuData;
            for (int on = 0; on < n; on++) {
                for (int om = 0; om < m; om++) {
                    int o = on * m + om;
                    for (int i = 0; i < spatial; i++) {
                        if (maskData[on * spatial + i] > 0.99) {
                            attnData[o * spatial + i] = maskValue;
                        }
                    }
                }
            }
        } else if (input.dataType == DataType::FLOAT16) {
            float *maskData = (float *) mask.cpuData;
            uint16_t *attnData = (uint16_t *) input.cpuData;
            uint16_t hMaskValue = float_to_half(maskValue);
            for (int on = 0; on < n; on++) {
                for (int om = 0; om < m; om++) {
                    int o = on * m + om;
                    for (int i = 0; i < spatial; i++) {
                        if (maskData[on * spatial + i] > 0.99) {
                            attnData[o * spatial + i] = hMaskValue;
                        }
                    }
                }
            }
        } else {
            ErrorInXLLM("AttentionMask error: unsupport input's dataType.\n");
        }
    }

    void SoftMax(Data &input, Data &output, int axis) {
        output.Allocate();

        AssertInXLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16,
                        "Softmax error: Data's type should be float32.\n");

        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int outer = input.Count(0) / input.Count(axis);
        int channels = input.dims[axis];
        int inner = input.Count(axis + 1);

        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;

        if (input.dataType == DataType::FLOAT16) {
            int len = input.Count(0);
            inputData = new float[len];
            outputData = new float[len];
            for (int i = 0; i < len; i++) {
                inputData[i] = fp16tofp32.dict[((uint16_t *) input.cpuData)[i]];
            }
        }

        if (inner == 1) {
            for (int i = 0; i < outer; i++) {
                float maxValue = 0;
                int j = 0;
                for (; j < channels; j++) {
                    maxValue = std::max(maxValue, inputData[j]);
                }

                j = 0;
                for (; j < channels; j++) {
                    outputData[j] = exp(inputData[j] - maxValue);
                }
                float sum = 0.0;
                j = 0;
                for (; j < channels; j++) {
                    sum += outputData[j];
                }
                if (fabs(sum) < 1e-9) {
                    sum = 0.1;
                }
                j = 0;
                for (; j < channels; j++) {
                    outputData[j] = outputData[j] / sum;
                }
                inputData += channels;
                outputData += channels;
            }
        } else {
            for (int i = 0; i < outer; i++) {
                std::vector<float> maxValue(inner, -FLT_MAX);
                for (int j = 0; j < channels; j++) {
                    for (int k = 0; k < inner; k++) {
                        maxValue[k] = std::max(maxValue[k], inputData[j * inner + k]);
                    }
                }
                std::vector<float> sum(inner, 0.0);
                for (int j = 0; j < channels; j++) {
                    for (int k = 0; k < inner; k++) {
                        outputData[j * inner + k] = std::exp(inputData[j * inner + k] - maxValue[k]);
                        sum[k] += outputData[j * inner + k];
                    }
                }

                for (int j = 0; j < channels; j++) {
                    for (int k = 0; k < inner; k++) {
                        outputData[j * inner + k] /= sum[k];
                    }
                }

                inputData += channels * inner;
                outputData += channels * inner;
            }
        }

        if (input.dataType == DataType::FLOAT16) {
            int len = input.Count(0);
            inputData -= len;
            outputData -= len;
            for (int i = 0; i < len; i++) {
                ((uint16_t *) output.cpuData)[i] = float_to_half(outputData[i]);
            }

            delete[] inputData;
            delete[] outputData;
        }
    }

    void MatMulSingle(float *input0Base, float *input1Base, float *outputBase,
                      int input0Spatial, int input1Spatial, int outputSpatial,
                      int n, int m, int k, float alpha, int st, int end) {
        for (int b = st; b < end; b++) {
            float *input0Data = input0Base + b * input0Spatial;
            float *input1Data = input1Base + b * input1Spatial;
            float *outputData = outputBase + b * outputSpatial;
            std::fill(outputData, outputData + n * k, 0.0f);
            // 一次只计算一个数，不需要SIMD
            // 遍历input0的行
            for (int i = 0; i < n; i++) {
                // 遍历input1的行
                for (int j = 0; j < m; j++) {
                    float now = input0Data[i * m + j] * alpha;
                    for (int l = 0; l < k; l++) {
                        outputData[i * k + l] += (now * input1Data[j * k + l]);
                    }
                }
            }
        }
    }

    // input0: {1, num_attention_heads, bsz * seqlen, k_seqlen}
    // input1:{num_attention_heads, k_seqlen, hidden_size/num_attention_heads}
    // output: {1, num_attention_heads, bsz * seqlen, hidden_size/num_attention_heads}
    // input0(batch,n,m) * input1(batch,m,k) = output(batch,n,k)
    // input1内存不连续，并且需要转置
    // 如果input1先permute，就跟MatMulTransB一样了
    void MatMul(Data &input0, Data &input1, Data &output) {
        output.Allocate();

        int input0Spatial = input0.Count(input0.dims.size() - 2);     // n*m
        int input1Spatial = input1.dims.back()*input1.expandDims[input1.dims.size() - 2];  // k*expandDims[1]
        int outputSpatial = output.Count(output.dims.size() - 2);    // n*k

        // int input0Stride = input0.strides[input0.dims.size() - 2];   // m
        // int input1Stride = input1.strides[input1.dims.size() - 2];   // k

        int n = input0.dims[input0.dims.size() - 2];
        int m = input0.dims.back();
        int k = input1.dims.back();
        int batch = input0.Count(0) / input0Spatial;

        int threadNum = GetThreads();
        if (batch * n * m * k < 64 * 4096) {
            threadNum = 1;
        }
        // threadNum = std::min(threadNum, 4);
        int per = batch / threadNum;   // 一个线程负责per个batch
        int cur = 0;
        auto pool = GetPool();
        std::vector <std::future <void> > futures;
        float alpha = 1.0;
        if (input0.dataType == DataType::FLOAT32) {
            for (int i = 0; i < threadNum - 1; i++) {
                int end = cur + per + (cur + per * (threadNum - i) < batch);
                futures.push_back(pool->enqueue(MatMulSingle,
                                               (float *) input0.cpuData, (float *) input1.cpuData,
                                               (float *) output.cpuData,
                                               input0Spatial, input1Spatial, outputSpatial,
                                               n, m, k, alpha, cur, end));
                cur = end;
            }
            MatMulSingle((float *) input0.cpuData, (float *) input1.cpuData, (float *) output.cpuData,
                         input0Spatial, input1Spatial, outputSpatial, n, m, k, alpha, cur, batch);
        } 
        for (int i = 0; i < futures.size(); i++) {
            futures[i].get();
        }
    }

    void AddTo(Data &input0, Data &input1) {
        float alpha = 1.0;

        AssertInXLLM(input0.dataType == DataType::FLOAT32 || input1.dataType == DataType::FLOAT16,
                        "AddTo error: Data's type should be float32 or float16.\n");
        AssertInXLLM(input0.dims == input1.dims, "AddTo error: input's shape should be same.\n");

        int len = input0.Count(0);

        if (input0.dataType == DataType::FLOAT32) {
            float *input0Data = (float *) input0.cpuData;
            float *input1Data = (float *) input1.cpuData;
            for (int i = 0; i < len; i++) {
                input0Data[i] += input1Data[i] * alpha;
            }
        } else if (input0.dataType == DataType::FLOAT16) {
            uint16_t *input0Data = (uint16_t *) input0.cpuData;
            uint16_t *input1Data = (uint16_t *) input1.cpuData;
            for (int i = 0; i < len; i++) {
                input0Data[i] = float_to_half(fp16tofp32.dict[input0Data[i]] + fp16tofp32.dict[input1Data[i]] * alpha);
            }
        }
    }

    void MulTo(Data &input0, Data &input1) {
        AssertInXLLM(input0.dims == input1.dims, "MulTo error: input's shape should be same.\n");

        float *input0Data = (float*)input0.cpuData;
        float *input1Data = (float*)input1.cpuData;

        int len = input0.Count(0);
        int inner = input1.Count(0);
        AssertInXLLM(len % inner == 0, "MulTo error: Data`s shape can`t perform MulTo operation.\n");
        int round = (len / inner);
        for (int j = 0; j < round; j++) {
            for (int i = 0; i < len; i++) {
               input0Data[i] *= input1Data[i];
            }
            input0Data += inner;
        }
    }

    void Silu(Data &input, Data &output) {
        output.Allocate();
        AssertInXLLM(input.dataType == DataType::FLOAT32, "Silu error: Data's type should be float32.\n");
        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;
        int len = input.Count(0);
        int i = 0;
        for (; i < len; i++) {
            float x = inputData[i];
            outputData[i] = x / (1.0 + expf(-x));
        }
    }

}
