#include "xllm.h"
#include "executor.h"

namespace xllm {
    static int threads = 4;
    static ThreadPool *xllmThreadPool = new ThreadPool(threads);
    std::map <std::string, int> defaultDeviceMap;
    Executor defaultExecutor;
    Executor *curExecutor = &defaultExecutor;

    void PrintInstructionInfo() {
        std::string avx = "OFF", avx2 = "OFF";
#ifdef __AVX__
        avx = "ON";
#endif
#ifdef __AVX2__
        avx2 = "ON";
#endif
        printf("AVX: %s\n", avx.c_str());
        printf("AVX2: %s\n", avx2.c_str());
    }

    void SetThreads(int t) {
        threads = t;
        if (xllmThreadPool) delete xllmThreadPool;
        xllmThreadPool = new ThreadPool(t);
    }

    int GetThreads() {
        return threads;
    }

    ThreadPool *GetPool() {
        return xllmThreadPool;
    }

    void Embedding(const Data &input, Data &weight, Data &output) {
        curExecutor->Run("Embedding", {
                {"input", (Data*)&input}, {"weight", &weight}, {"output", &output}
        }, {}, {});
    }

    void RMSNorm(const Data &input, const Data &weight, Data &output, float eps) {
        curExecutor->Run("RMSNorm", {
                {"input", (Data*)&input}, {"weight", (Data*)&weight}, {"output", &output}
        }, {{"eps", eps}}, {});
    }
    
    void Linear(Data &input, Data &weight, Data &output) {
        curExecutor->Run("Linear", {
                {"input", &input}, {"weight", &weight}, {"output", &output}
        }, {}, {});
    }

    void LlamaRotatePosition2D(Data &input, const Data &positionIds, Data &sinData, Data &cosData, int rotaryDim) {
        curExecutor->Run("LlamaRotatePosition2D", {
                {"input", &input}, {"positionIds", (Data*)&positionIds}, {"sin", &sinData}, {"cos", &cosData}
        }, {}, {{"rotaryDim", rotaryDim}});
    }

    void PermuteSelf(const Data &input, const std::vector<int> &axis) {
        Data axisData = Data(DataType::INT32PARAM, {(int)axis.size()});
        axisData.Allocate();
        for (int i = 0; i < axisData.Count(0); i++) {
            ((int32_t*)axisData.cpuData)[i] = axis[i];
        }
        curExecutor->Run("PermuteSelf", {
                {"input", (Data*)&input}, {"axis", &axisData}
        }, {}, {});
    }

    void CatDirect(Data &input0, const Data &input1, int axis) {
        curExecutor->Run("CatDirect", {
                {"input0", (Data*)&input0}, {"input1", (Data*)&input1}
        }, {}, {{"axis", axis}});
    }

    void CatDirectFP16(Data &input0, const Data &input1, int axis) {
        curExecutor->Run("CatDirectFP16", {
                {"input0", (Data*)&input0}, {"input1", (Data*)&input1}
        }, {}, {{"axis", axis}});
    }

    void MatMulTransBFP16(const Data &input0, const Data &input1, Data &output, float alpha) {
        curExecutor->Run("MatMulTransBFP16", {
                {"input0", (Data*)&input0}, {"input1", (Data*)&input1}, {"output", &output}
        }, {{"alpha", alpha}}, {});
    }

    void MatMulTransB(const Data &input0, const Data &input1, Data &output, float alpha) {
        curExecutor->Run("MatMulTransB", {
                {"input0", (Data*)&input0}, {"input1", (Data*)&input1}, {"output", &output}
        }, {{"alpha", alpha}}, {});
    }

    void AttentionMask(Data &input, const Data &mask, float maskValue) {
        curExecutor->Run("AttentionMask", {
                {"input", &input}, {"mask", (Data*)&mask}
        }, {{"maskValue", maskValue}}, {});
    }

    void SoftMax(const Data &input, Data &output, int axis) {
        curExecutor->Run("SoftMax", {
                {"input", (Data*)&input}, {"output", &output}
        }, {}, {{"axis", axis}});
    }

    void MatMul(const Data &input0, const Data &input1, Data &output, float alpha) {
        curExecutor->Run("MatMul", {
                {"input0", (Data*)&input0}, {"input1", (Data*)&input1}, {"output", &output}
        }, {{"alpha", alpha}}, {});
    }

    void MatMulFP16(const Data &input0, const Data &input1, Data &output, float alpha) {
        curExecutor->Run("MatMulFP16", {
                {"input0", (Data*)&input0}, {"input1", (Data*)&input1}, {"output", &output}
        }, {{"alpha", alpha}}, {});
    }

    void AddTo(Data &input0, const Data &input1, float alpha) {
        curExecutor->Run("AddTo", {
                {"input0", &input0}, {"input1", (Data*)&input1}
        }, {{"alpha", alpha}}, {});
    }

    void Silu(const Data &input, Data &output) {
        curExecutor->Run("Silu", {
                {"input", (Data*)&input}, {"output", &output}
        }, {}, {});
    }

    void MulTo(Data &input0, const Data &input1) {
        curExecutor->Run("MulTo", {
                {"input0", &input0}, {"input1", (Data*)&input1}
        }, {}, {});
    }

    void ClearProfiler() {
        curExecutor->ClearProfiler();
    }

    void PrintProfiler() {
        curExecutor->PrintProfiler();
    }

    std::map <std::string, int> GetDeviceMap() {
        return defaultDeviceMap;
    }

    void ApplyDeviceMap(const std::map <std::string, int> &deviceMap, int current, int total) {
        if (deviceMap.size() == 0) {
            return;
        }
        int sum = 0, cur = 0;
        for (auto &it : deviceMap) {
            sum += it.second;
        }
        std::string curDevice = deviceMap.begin()->first;
        for (auto &it : deviceMap) {
            cur += it.second;
            // current / total <= cur / sum
            if (current * sum <= cur * total) {
                curDevice = it.first;
                break;
            }
        }
    }

    void TopK(const Data &input, Data &output, int topk) {
        curExecutor->Run("TopK", {
                {"input", (Data*)&input}, {"output", &output}
        }, {}, {{"topk", topk}});
    };

    void Split(const Data &input, int axis, int start, int end, Data &output) {
        curExecutor->Run("Split", {
                {"input", (Data*)&input}, {"output", &output}
        }, {}, {{"axis", axis}, {"start", start}, {"end", end}});
    }

    void Attention(const Data &q, const Data &k, const Data &v, const Data &mask, Data &output,
                   int group, float scale, int attentionType) {
        curExecutor->Run("Attention", {
                {"q", (Data*)&q}, {"k", (Data*)&k}, {"v", (Data*)&v},
                {"mask", (Data*)&mask}, {"output", (Data*)&output}
        }, {{"scale", scale}}, {{"group", group}});
    }
}