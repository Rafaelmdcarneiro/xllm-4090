#pragma once

#include <string>
#include <iostream>
#include <vector>
#include <math.h>
#include <map>

#include "threadpool.h"
#include "data.h"

namespace xllm {
    void PrintInstructionInfo();
    void SetThreads(int t);
    int GetThreads();
    ThreadPool *GetPool();
    void PrintProfiler();
    std::map <std::string, int> GetDeviceMap();  // 用于多卡部署
    void ApplyDeviceMap(const std::map <std::string, int> &deviceMap, int current, int total);

    class Data;

    struct GenerationConfig {
        int output_token_limit = -1; // 最多输出多少, <= 0代表无限制
        int last_n = 64; // 末尾last_n个token计入重复惩罚
        float repeat_penalty = 1.0f; // 重复惩罚系数，1.0代表不惩罚
        int top_k = 1; // top_k采样
        float top_p = 0.9; // top_p采样
        float temperature = 0.6; // 温度参数，一般在0.1 ~ 1.0之间，设大这个参数可以带来结果的多样性

        bool IsSimpleGreedy() const {
            if (fabs(repeat_penalty - 1) > 1e-8) {
                return false;
            }
            if (top_k > 1) {
                return false;
            }
            return true;
        }
    };

    void Embedding(const Data &input, Data &weight, Data &output);
    void RMSNorm(const Data &input, const Data &weight, Data &output, float eps);
    void Linear(Data &input, Data &weight, Data &output);
    void LlamaRotatePosition2D(Data &input, const Data &positionIds, Data &sinData, Data &cosData, int rotaryDim); // 2D position for llama
    void PermuteSelf(const Data &input, const std::vector<int> &axis); // 转置
    void CatDirect(Data &input0, const Data &input1, int axis); // 直接把input1的数据拷贝到input0后面（需要input0提前扩容了足够的空间）
    void CatDirectFP16(Data &input0, const Data &input1, int axis); 
    void MatMulTransB(const Data &input0, const Data &input1, Data &output, float alpha = 1.0);
    void MatMulTransBFP16(const Data &input0, const Data &input1, Data &output, float alpha = 1.0);
    void AttentionMask(Data &input, const Data &mask, float maskValue); // 把input里对应位置mask中为1的部分变成maskValue
    void SoftMax(const Data &input, Data &output, int axis);
    void MatMul(const Data &input0, const Data &input1, Data &output, float alpha = 1.0);
    void MatMulFP16(const Data &input0, const Data &input1, Data &output, float alpha = 1.0);
    void AddTo(Data &input0, const Data &input1, float alpha = 1.0); // input0 += input1 * alpha
    void Silu(const Data &input, Data &output);
    void MulTo(Data &input0, const Data &input1); // input0 *= input1
    void TopK(const Data &input, Data &output, int topk);
    void Split(const Data &input, int axis, int start, int end, Data &output);
    void Attention(const Data &q, const Data &k, const Data &v, const Data &mask, Data &output,
                   int group, float scale, int attentionType);
}