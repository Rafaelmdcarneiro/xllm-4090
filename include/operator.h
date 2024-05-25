#pragma once

#include "data.h"


namespace xllm{

    typedef std::map <std::string, float> FloatDict;
    typedef std::map <std::string, int> IntDict;
    
    void Embedding(const Data &input, Data &weight, Data &output);

    void RMSNorm(const Data &input, Data &weight, Data &output,float eps);

    void Linear(const Data &input, Data &weight, Data &output);

    void LlamaRotatePosition2D(Data &input, const Data &positionIds, Data &sinData, Data &cosData, int rotaryDim); // 2D position for llama

    void PermuteSelf(Data &input, std::vector <int> axis);

    void CatDirect(Data &input0, Data &input1, int axis);

    void MatMulTransB(Data &input0, Data &input1, Data &output, float alpha);

    void AttentionMask(Data &input, const Data &mask, float maskValue);

    void SoftMax(Data &input, Data &output, int axis);

    void MatMul(Data &input0, Data &input1, Data &output);

    void AddTo(Data &input0, Data &input1);

    void MulTo(Data &input0, Data &input1);

    void Silu(Data &input, Data &output);

    void MultiplyMultiThread(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k, int threadNum);
}
