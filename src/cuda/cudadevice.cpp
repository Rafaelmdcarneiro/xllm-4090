#include "cuda/cudadevice.h"
#include "cuda/xllm-cuda.cuh"

namespace xllm {
    CudaDevice::CudaDevice() {
        this->deviceType = "cuda";
        this->ops["RMSNorm"] = (BaseOperator*)(new CudaRMSNormOp());
        this->ops["Linear"] = (BaseOperator*)(new CudaLinearOp());
        this->ops["LlamaRotatePosition2D"] = (BaseOperator*)(new CudaLlamaRotatePosition2DOp());
        this->ops["PermuteSelf"] = (BaseOperator*)(new CudaPermuteSelfOp());
        this->ops["CatDirect"] = (BaseOperator*)(new CudaCatDirectOp());
        this->ops["CatDirectFP16"] = (BaseOperator*)(new CudaCatDirectFP16Op());
        this->ops["MatMulTransB"] = (BaseOperator*)(new CudaMatMulTransBOp());
        this->ops["MatMulTransBFP16"] = (BaseOperator*)(new CudaMatMulTransBFP16Op());
        this->ops["AttentionMask"] = (BaseOperator*)(new CudaAttentionMaskOp());
        this->ops["SoftMax"] = (BaseOperator*)(new CudaSoftMaxOp());
        this->ops["MatMul"] = (BaseOperator*)(new CudaMatMulOp());
        this->ops["MatMulFP16"] = (BaseOperator*)(new CudaMatMulFP16Op());
        this->ops["AddTo"] = (BaseOperator*)(new CudaAddToOp());
        this->ops["Silu"] = (BaseOperator*)(new CudaSiluOp());
        this->ops["MulTo"] = (BaseOperator*)(new CudaMulToOp());
        this->ops["TopK"] = (BaseOperator*)(new CudaTopKOp());
        this->ops["Split"] = (BaseOperator*)(new CudaSplitOp());
        this->ops["Attention"] = (BaseOperator*)(new CudaAttentionOp());
    }

    bool CudaDevice::Malloc(void **ret, size_t size) {
        *ret = xllmCudaMalloc(size);
        return true;
    }

    bool CudaDevice::Free(void *ret) {
        xllmCudaFree(ret);
        return true;
    }

    bool CudaDevice::CopyDataFromCPU(void *dst, void *src, size_t size) {
        xllmCudaCopyFromHostToDevice(dst, src, size);
        return true;
    }

    bool CudaDevice::CopyDataToCPU(void *dst, void *src, size_t size) {
        xllmCudaCopyFromDeviceToHost(dst, src, size);
        return true;
    }

    void CudaRMSNormOp::Run(const std::string &opType, const DataDict &datas,
                       const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();

        float eps = floatParams.find("eps") != floatParams.end() ? floatParams.find("eps")->second : 1e-5;
        xllmCudaRMSNorm(input, weight, output, eps);
    }

    void CudaLinearOp::Run(const std::string &opType, const DataDict &datas,
                           const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &bias = *(datas.find("bias")->second);

        output.Allocate();
        int n = input.Count(0) / input.dims.back();
        int m = input.dims.back();
        int k = output.dims.back();

        if (weight.dataType == DataType::FLOAT16) {
            xllmCudaMatMulFloat16(input, weight, bias, output, n, m, k);
        } else {
            ErrorInXLLM("Linear error: unsupport weight's dataType.\n");
        }
    }

    void CudaCatDirectFP16Op::Run(const std::string &opType, const DataDict &datas,
                             const FloatDict &floatParams, const IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);

        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;

        // AssertInXLLM(input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32,
        //                 "Cat's input's type should be float32.\n");
        AssertInXLLM(input0.dataDevice == input1.dataDevice, "CatDirect error: inputs should use same device.\n");

        if (input0.dims.size() == 0) {
            input0.assignBytes = input1.assignBytes/2;
            input0.counts = input1.counts;
            input0.Resize(input1.dims);
            AssertInXLLM(input0.expandDims.size() == input1.dims.size() &&
                            input1.dims[axis] <= input0.expandDims[axis],
                            "CatDirect Error: input0's expansion size is not enough.\n");
            int outer = input1.Count(0) / input1.Count(axis);
            int input0Stride = input0.Count(axis) * input0.expandDims[axis]/input0.dims[axis];
            int input1Stride = input1.Count(axis);
            int inner = input0.strides[axis];
            int unitSize = input0.unitSize;
            xllmCudaMemcpy2DDeviceToDeviceFP16((uint8_t *) input0.cudaData, input0Stride * unitSize,
                                              (uint8_t *) input1.cudaData, input1Stride * unitSize,
                                              input1.dims[axis] * inner * unitSize, outer);
            return;
        }

        AssertInXLLM(input0.dims.size() == input1.dims.size(), "Cat Error: input's shape's size should be same.\n");
        int dimsLen = input0.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;

        for (int i = 0; i < dimsLen; i++) {
            if (i != axis) {
                AssertInXLLM(input0.dims[i] == input1.dims[i], "Cat Error: input's shape doesn't match.");
            }
        }

        input0.assignBytes += input1.assignBytes/2;
        input0.counts += input1.counts;
        std::vector<int> dims = input0.dims;
        std::vector<int> oldDims = dims;
        dims[axis] += input1.dims[axis];
        input0.Resize(dims);
        int outer = input0.Count(0) / input0.Count(axis);
        int input0Stride = input0.expandDims[axis] * input0.expandDims[axis+1];
        int input1Stride = input1.Count(axis);

        int inner = input0.strides[axis];
        int unitSize = input0.unitSize;

        xllmCudaMemcpy2DDeviceToDeviceFP16((uint8_t *) input0.cudaData + oldDims[axis] * inner * unitSize,
                                          input0Stride * unitSize,
                                          (uint8_t *) input1.cudaData, input1Stride * unitSize,
                                          input1.dims[axis] * inner * unitSize, outer);
    }

    void CudaCatDirectOp::Run(const std::string &opType, const DataDict &datas,
                             const FloatDict &floatParams, const IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);

        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;

        AssertInXLLM(input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32,
                        "Cat's input's type should be float32.\n");
        AssertInXLLM(input0.dataDevice == input1.dataDevice, "CatDirect error: inputs should use same device.\n");

        if (input0.dims.size() == 0) {
            input0.assignBytes = input1.assignBytes;
            input0.counts = input1.counts;
            input0.Resize(input1.dims);
            AssertInXLLM(input0.expandDims.size() == input1.dims.size() &&
                            input1.dims[axis] <= input0.expandDims[axis],
                            "CatDirect Error: input0's expansion size is not enough.\n");
            int outer = input1.Count(0) / input1.Count(axis);
            int input0Stride = input0.Count(axis) * input0.expandDims[axis]/input0.dims[axis];
            int input1Stride = input1.Count(axis);
            int inner = input0.strides[axis];
            int unitSize = input0.unitSize;
            xllmCudaMemcpy2DDeviceToDevice((uint8_t *) input0.cudaData, input0Stride * unitSize,
                                              (uint8_t *) input1.cudaData, input1Stride * unitSize,
                                              input1.dims[axis] * inner * unitSize, outer);
            return;
        }

        AssertInXLLM(input0.dims.size() == input1.dims.size(), "Cat Error: input's shape's size should be same.\n");
        int dimsLen = input0.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;

        for (int i = 0; i < dimsLen; i++) {
            if (i != axis) {
                AssertInXLLM(input0.dims[i] == input1.dims[i], "Cat Error: input's shape doesn't match.");
            }
        }

        input0.assignBytes += input1.assignBytes;
        input0.counts += input1.counts;
        std::vector<int> dims = input0.dims;
        std::vector<int> oldDims = dims;
        dims[axis] += input1.dims[axis];
        input0.Resize(dims);
        int outer = input0.Count(0) / input0.Count(axis);
        int input0Stride = input0.expandDims[axis] * input0.expandDims[axis+1];
        int input1Stride = input1.Count(axis);

        int inner = input0.strides[axis];
        int unitSize = input0.unitSize;

        xllmCudaMemcpy2DDeviceToDevice((uint8_t *) input0.cudaData + oldDims[axis] * inner * unitSize,
                                          input0Stride * unitSize,
                                          (uint8_t *) input1.cudaData, input1Stride * unitSize,
                                          input1.dims[axis] * inner * unitSize, outer);
    }

    void CudaMatMulOp::Run(const std::string &opType, const DataDict &datas,
                          const FloatDict &floatParams, const IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        Data &output = *(datas.find("output")->second);

        output.Allocate();

        float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : -1;
        int input0Spatial = input0.Count(input0.dims.size() - 2);
        int input1Spatial = input1.dims.back()*input1.expandDims[input1.dims.size() - 2];
        int input0Stride = input0.strides[input0.dims.size() - 2];
        int input1Stride = input1.strides[input1.dims.size() - 2];
        int n = input0.dims[input0.dims.size() - 2];
        int m = input0.dims.back();
        int k = input1.dims[input1.dims.size() - 1];
        int batch = input0.Count(0) / input0Spatial;

        int outputSpatial = output.Count(output.dims.size() - 2);
        xllmCudaBatchMatMul(input0, input1, output,
                     input0Spatial, input1Spatial, outputSpatial, input0Stride, input1Stride,
                     batch, n, m, k, alpha);
    }

    void CudaMatMulFP16Op::Run(const std::string &opType, const DataDict &datas,
                          const FloatDict &floatParams, const IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        Data &output = *(datas.find("output")->second);

        output.Allocate();

        float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : -1;
        int input0Spatial = input0.Count(input0.dims.size() - 2);
        int input1Spatial = input1.dims.back()*input1.expandDims[input1.dims.size() - 2];
        int input0Stride = input0.strides[input0.dims.size() - 2];
        int input1Stride = input1.strides[input1.dims.size() - 2];
        int n = input0.dims[input0.dims.size() - 2];
        int m = input0.dims.back();
        int k = input1.dims[input1.dims.size() - 1];
        int batch = input0.Count(0) / input0Spatial;

        int outputSpatial = output.Count(output.dims.size() - 2);
        xllmCudaBatchMatMulFP16(input0, input1, output,
                     input0Spatial, input1Spatial, outputSpatial, input0Stride, input1Stride,
                     batch, n, m, k, alpha);
    }
    
    void CudaMatMulTransBOp::Run(const std::string &opType, const DataDict &datas,
                                const FloatDict &floatParams, const IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        Data &output = *(datas.find("output")->second);

        output.Allocate();

        float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : -1;
        int input0Spatial = input0.Count(input0.dims.size() - 2);
        int input1Spatial = input1.dims.back() * input1.expandDims[input1.dims.size() - 2];
        int input0Stride = input0.strides[input0.dims.size() - 2];
        int input1Stride = input1.strides[input1.dims.size() - 2];
        int n = input0.dims[input0.dims.size() - 2];
        int m = input0.dims.back();
        int k = input1.dims[input1.dims.size() - 2];
        int batch = input0.Count(0) / input0Spatial;

        int outputSpatial = output.Count(output.dims.size() - 2);
        xllmCudaBatchMatMulTransB(input0, input1, output,
                     input0Spatial, input1Spatial, outputSpatial, input0Stride, input1Stride,
                     batch, n, m, k, alpha);
    }

    void CudaMatMulTransBFP16Op::Run(const std::string &opType, const DataDict &datas,
                                const FloatDict &floatParams, const IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        Data &output = *(datas.find("output")->second);

        output.Allocate();

        float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : -1;
        int input0Spatial = input0.Count(input0.dims.size() - 2);
        int input1Spatial = input1.dims.back() * input1.expandDims[input1.dims.size() - 2];
        int input0Stride = input0.strides[input0.dims.size() - 2];
        int input1Stride = input1.strides[input1.dims.size() - 2];
        int n = input0.dims[input0.dims.size() - 2];
        int m = input0.dims.back();
        int k = input1.dims[input1.dims.size() - 2];
        int batch0 = input0.Count(0) / input0Spatial;
        int batch1 = input1.Count(0) / input1Spatial;

        int outputSpatial = output.Count(output.dims.size() - 2);
        xllmCudaBatchMatMulTransBFP16(input0, input1, output,
                     input0Spatial, input1Spatial, outputSpatial, input0Stride, input1Stride,
                     batch0, n, m, k, alpha);
    }

    bool CudaSoftMaxOp::CanRun(const std::string &opType, const DataDict &datas,
                               const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int inner = input.Count(axis + 1);
        if (inner != 1) {
            return false;
        }
        return true;
    }

    void CudaSoftMaxOp::Run(const std::string &opType, const DataDict &datas,
                            const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();

        AssertInXLLM(input.dataType == DataType::FLOAT32, "Softmax error: Data's type should be float32.\n");
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        xllmCudaSoftmax(input, output, axis);
    }

    void CudaSiluOp::Run(const std::string &opType, const DataDict &datas,
                            const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        AssertInXLLM(input.dataType == DataType::FLOAT32, "Silu error: Data's type should be float32.\n");
        xllmCudaSilu(input, output);
    }

    void CudaAddToOp::Run(const std::string &opType, const DataDict &datas,
                         const FloatDict &floatParams, const IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : 1.0;

        AssertInXLLM(input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32,
                        "AddTo error: Data's type should be float32.\n");
        AssertInXLLM(input0.dims == input1.dims, "AddTo error: input's shape should be same.\n");
        xllmCudaAddTo(input0, input1, alpha);
    }

    void CudaMulToOp::Run(const std::string &opType, const DataDict &datas,
                          const FloatDict &floatParams, const IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : 1.0;

        AssertInXLLM(input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32,
                        "MulTo error: Data's type should be float32.\n");
        AssertInXLLM(input0.dims == input1.dims, "MulTo error: input's shape should be same.\n");
        xllmCudaMulTo(input0, input1, alpha);
    }

    void CudaAttentionMaskOp::Run(const std::string &opType, const DataDict &datas,
                                  const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &mask = *(datas.find("mask")->second);
        float maskValue = floatParams.find("maskValue") != floatParams.end() ? floatParams.find("maskValue")->second : -10000.0;
        xllmCudaAttentionMask(input, mask, maskValue);
    }

    void CudaPermuteSelfOp::Run(const std::string &opType, const DataDict &datas,
                               const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &axisData = *(datas.find("axis")->second);
        std::vector <int> axis;
        for (int i = 0; i < axisData.Count(0); i++) {
            axis.push_back(((int32_t *) axisData.cpuData)[i]);
        }

        AssertInXLLM(input.dataType == DataType::FLOAT32, "Permute error: datatype should be float32.");
        AssertInXLLM(axis.size() == input.dims.size(), "Permute error: axis's size should be equal to data's shape's size.");

        bool same = false;
        same |= ((axis == std::vector <int>{1, 2, 0} || axis == std::vector <int>{1, 0, 2}) && (input.dims[0] == 1 || input.dims[1] == 1));
        same |= ((axis == std::vector <int>{2, 0, 1, 3}) && input.dims[2] == 1);
        same |= ((axis == std::vector <int>{0, 2, 1, 3}) && (input.dims[1] == 1 || input.dims[2] == 1));
        if (same) {
            std::vector<int> new_dims;
            for (int i = 0; i < axis.size(); i++) {
                new_dims.push_back(input.dims[axis[i]]);
            }
            input.Resize(new_dims);
            return;
        }

        xllmCudaPermute(input, axis);
    }

    void CudaLlamaRotatePosition2DOp::Run(const std::string &opType, const DataDict &datas,
                                     const FloatDict &floatParams, const IntDict &intParams) {
        Data &data = *(datas.find("input")->second);
        Data &positionIds = *(datas.find("positionIds")->second);
        Data &sinData = *(datas.find("sin")->second);
        Data &cosData = *(datas.find("cos")->second);
        int rotaryDim = intParams.find("rotaryDim") != intParams.end() ? intParams.find("rotaryDim")->second : 128;

        xllmCudaLlamaRotatePosition2D(data, positionIds, sinData, cosData, rotaryDim);
    }

    bool CudaTopKOp::CanRun(const std::string &opType, const xllm::DataDict &datas,
                            const xllm::FloatDict &floatParams, const xllm::IntDict &intParams) {
        int topk = intParams.find("topk") != intParams.end() ? intParams.find("topk")->second : 1;
        if (topk != 1) {
            return false;
        }
        return true;
    }

    void CudaTopKOp::Run(const std::string &opType, const xllm::DataDict &datas,
                        const xllm::FloatDict &floatParams, const xllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        int topk = intParams.find("topk") != intParams.end() ? intParams.find("topk")->second : -1;
        xllmCudaTopK(input, output, topk);
    }

    void CudaSplitOp::Run(const std::string &opType, const xllm::DataDict &datas,
                          const xllm::FloatDict &floatParams, const xllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);

        output.Allocate();

        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int start = intParams.find("start") != intParams.end() ? intParams.find("start")->second : 0;
        int end = intParams.find("end") != intParams.end() ? intParams.find("end")->second : 0;

        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        start = std::max(0, std::min(input.dims[axis] - 1, start));
        end = std::max(0, std::min(input.dims[axis], end));

        int outer = input.Count(0) / input.Count(axis);
        int inputStride = input.Count(axis);
        int outputStride = output.Count(axis);
        int channels = input.dims[axis];
        int inner = input.strides[axis];
        int unitSize = input.unitSize;

        xllmCudaMemcpy2DDeviceToDevice((uint8_t*)output.cudaData, outputStride * unitSize,
                                          (uint8_t*)input.cudaData + start * inner * unitSize, inputStride * unitSize,
                                          (end - start) * inner * unitSize, outer);
    }

    void CudaAttentionOp::Run(const std::string &opType, const xllm::DataDict &datas,
                           const xllm::FloatDict &floatParams, const xllm::IntDict &intParams) {
        Data emptyData;
        Data &q = *(datas.find("q")->second);
        Data &k = *(datas.find("k")->second);
        Data &v = *(datas.find("v")->second);
        Data &mask = datas.find("mask")->second ? *(datas.find("mask")->second) : emptyData;
        Data &output = *(datas.find("output")->second);
        int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : 1;
        float scale = floatParams.find("scale") != floatParams.end() ? floatParams.find("scale")->second : 1.0;
        output.Allocate();
        xllmCudaAttention(q, k, v, mask, output, group, scale);
    }
}
