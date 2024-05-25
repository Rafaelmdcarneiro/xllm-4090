#include "executor.h"
#include "cpu/cpudevice.h"

#ifdef USE_CUDA
#include "cuda/cudadevice.h"
#endif

namespace xllm {
    Executor::Executor() {
        this->devices.clear();
#ifdef USE_CUDA
        this->devices.push_back((BaseDevice*) new CudaDevice());
#endif
        this->devices.push_back((BaseDevice*) new CpuDevice());
    }

    Executor::~Executor() {
        for (int i = 0; i < devices.size(); i++) {
            delete devices[i];
        }
    }

    void Executor::Run(const std::string &opType, const xllm::DataDict &datas, const xllm::FloatDict &floatParams,
                       const xllm::IntDict &intParams) {
        auto st = std::chrono::system_clock::now();
        bool lockInCPU = false;

        for (auto device: devices) {
            if (device->CanRun(opType, datas, floatParams, intParams)) {
                for (auto &it: datas) {
                    it.second->ToDevice((void *) device);
                }
                device->Run(opType, datas, floatParams, intParams);
                break;
            }
        }
        float spend = GetSpan(st, std::chrono::system_clock::now());
        profiler[opType] += spend;
    }

    void Executor::ClearProfiler() {
        profiler.clear();
    }

    void Executor::PrintProfiler() {
        float sum = 0.0;
        for (auto &it : profiler) {
            printf("%s spend %f\n", it.first.c_str(), it.second);
            sum += it.second;
        }
        printf("total spend %f\n", sum);
    }
}