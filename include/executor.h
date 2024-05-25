#pragma once

#include "device.h"

namespace xllm {
    class Executor {
    private:
        std::vector <BaseDevice*> devices;
        std::map <std::string, float> profiler;

    public:
        Executor ();

        ~Executor();

        // 运行一个op
        void Run(const std::string &opType, const xllm::DataDict &datas, const xllm::FloatDict &floatParams,
                 const xllm::IntDict &intParams);

        void ClearProfiler();

        void PrintProfiler();
    };
}