#pragma once
#include <map>
#include <cstring>
#include <cmath>

#include "file.h"
#include "utils.h"
#include "xllm.h"

namespace xllm{

enum DataType {
    FLOAT32 = 0, BFLOAT16 = 1, INT8 = 3, INT4 = 4, INT4_NOZERO = 8, FLOAT16 = 7,
    INT32PARAM = 100 // int32的参数，这种类型的数据永远存在CPU上
};

enum WeightType {
    NONE = 0, LINEAR = 1, EMBEDDING = 2
};

enum DataDevice {
    CPU = 0, CUDA = 1
};

struct LowBitConfig {
    int bit;
    float min, max;   // 浮点数的最小值、最大值
    uint8_t zeroPoint;
    float scale;
    int type; // 0: 非对称量化 1: 对称量化

    LowBitConfig(float min, float max, int bit, int type) {
        this->min = min;
        this->max = max;
        this->bit = bit;
        this->type = type;
        Reset();
    }

    LowBitConfig () {

    }

    void Reset() {
        /*if (type == 1) {
            this->scale = (max - min) / 15.0;
            return;
        }*/
        /*if (type == 1) {
            this->scale = std::max(fabs(max), fabs(min)) / 7.0;
            this->min = this->scale * (-7.0);
            return;
        }*/
        min = std::min(min, 0.f);
        max = std::max(max, 0.f);

        const float qmin = 0;
        const float qmax = (1 << bit) - 1;
        scale = (max - min) / (qmax - qmin);
        const float initial_zero_point = qmin - min / scale;
        zeroPoint = 0;
        if (initial_zero_point < qmin) {
            zeroPoint = qmin;
        } else if (initial_zero_point > qmax) {
            zeroPoint = qmax;
        } else {
            zeroPoint = static_cast<uint8_t>(std::round(initial_zero_point));
        }

        if (type == 1) {
            this->min = -this->scale * zeroPoint;
            return;
        }
    }

    uint8_t quantization(const float &realNumber) const {
        if (type == 0) {
            return (uint8_t) (std::min((double) ((1 << bit) - 1),
                                        std::max(realNumber / scale + zeroPoint + 0.5, 0.0)));
        } else {
            return (uint8_t) (std::max(0.f, std::min(15.f, (realNumber - min) / scale + 0.5f)));
        }
    }

    float invQuantization(const uint8_t &qNumber) const {
        if (type == 0) {
            return (scale * ((float) qNumber - (float) zeroPoint));
        } else {
            return min + scale * qNumber;
        }
    }
};

class Data {
    public:
        WeightType weightType = WeightType::NONE; // 权重类型，NONE代表非Embedding/Linear

        DataType dataType = DataType::FLOAT32; // 数据类型
        int unitSize = 4;         // dataType占几个字节
        int unitSizeDiv = 1;  // unitSIze / unitSizeDiv: 单个元素占几个字节

        // 真实数据信息
        std::vector <int> dims; // 形状
        int counts = 0; // 数量
        uint64_t bytes = 0; // 字节数
        // 跨度, 用于快速计算指定维度的元素数量
        // 维度与dims相同，strides[i]=dims[i+1]*dims[i+2]*...*1
        // 例如：dims(2,3,4) -> strides(12,4,1), dims(2,3) -> strides(3,1)
        std::vector <uint64_t> strides;
        uint64_t assignBytes = 0; // 字节

        // 扩张空间信息 (>=真实数据)
        std::vector <int> expandDims;  // 形状
        uint64_t expandCounts = 0; // 数量
        uint64_t expandBytes = 0; // 字节

        uint8_t *cpuData = nullptr; // 数据指针
        void *cudaData = nullptr;
        std::vector <void*> extraCudaData;  //

        DataDevice dataDevice = DataDevice::CPU;  // 标记当前数据在哪个设备上
        std::vector <int> dataDeviceIds;
        void *deviceData = nullptr;

        // 量化相关的参数
        int perChannelAxis = -1; // 沿哪个轴分通道量化，-1代表没有分通道
        std::vector <LowBitConfig> perChannelsConfigs; // perChannelsConfigs[i]代表第i个通道的min, max; 如果没有分通道，perChannelsConfigs[0]代表全局min, max
        std::vector <float> scales;
        std::vector <int> zeros;
        std::vector <int> weightSum; // 作为权重时，有时候需要存一些和加速计算

        Data() {};
        Data (DataType type);
        Data (DataType type, const std::vector <int> &dims);
        Data (DataType type, const std::vector <int> &dims, const std::vector <float> &data);

        void ToDevice(DataDevice device);
        void ToDevice(void *device);
        void ToDevice(DataDevice device, const std::vector <int> &deviceIds);
        
        ~Data(); 

        Data (const Data &ori); // 深拷贝
        void CopyFrom(const Data &ori); // 复制
        
        void Allocate(); // 分配内存
        void MallocSpace(uint64_t bytes); // 在设备上分配
        void FreeSpace(); // 回收设备上的内存
        void Allocate(float v);

        void UpdateUnitSize(); // 更新unitSize

        void Resize(const std::vector <int> &dims); // 更改尺寸
        void Reshape(const std::vector <int> &dims); // 更改尺寸, 但不移动数据

        uint64_t Count(int i) const;

        void Expansion(const std::vector <int> &dims); // 预扩容到相应尺寸

        void PrintShape() const; // 输出形状
        void Print() const; // 输出

        void CalcWeightSum(); // 计算WeightSum

        void removeBatch(std::vector<int> removedBatch, int batch);
    };
}
