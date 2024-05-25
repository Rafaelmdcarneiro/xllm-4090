#include "device.h"

namespace xllm {

    bool BaseDevice::Malloc(void **ret, Data &data) {
        return Malloc(ret, data.expandBytes);
    }

    bool BaseDevice::CopyDataFromCPU(Data &data) {
        AssertInXLLM(data.cpuData != nullptr, "Copy data to " + this->deviceName + " from cpu failed: cpu's data is null.\n");
        AssertInXLLM(data.deviceData == nullptr, "Copy data to " + this->deviceName + " from cpu failed: device's data is not null.\n");
        Malloc(&data.deviceData, data.expandBytes);
        bool ret = CopyDataFromCPU(data.cudaData, data.cpuData, data.expandBytes);
        delete[] data.cpuData;
        data.cpuData = nullptr;
        return ret;
    }

    bool BaseDevice::CopyDataToCPU(Data &data) {
        AssertInXLLM(data.cpuData == nullptr, "Copy data from " + this->deviceName + " to cpu failed: cpu's data is not null.\n");
        AssertInXLLM(data.deviceData != nullptr, "Copy data from " + this->deviceName + " to cpu failed: device's data is null.\n");
        data.cpuData = new uint8_t [data.expandBytes];
        bool ret = CopyDataToCPU(data.cpuData, data.deviceData, data.expandBytes);
        this->Free(data.deviceData);
        data.deviceData = nullptr;
        return ret;
    }

    bool BaseOperator::CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams,
                              const IntDict &intParams) {
        return true;
    }

    bool BaseDevice::CanRun(const std::string &opType, const xllm::DataDict &datas,
                            const xllm::FloatDict &floatParams, const xllm::IntDict &intParams) {
        if (this->ops.find(opType) == this->ops.end()) {
            return false;
        }
        return this->ops[opType]->CanRun(opType, datas, floatParams, intParams);
    }

    void BaseDevice::Run(const std::string &opType, const DataDict &datas,
                         const FloatDict &floatParams, const IntDict &intParams) {
        this->ops[opType]->Run(opType, datas, floatParams, intParams);
    }
}