#include <gtest/gtest.h>
#include <stdio.h>
#include "data.h"
#include "operator.h"
#include "param.h"

using namespace xllm;

TEST(test_data, CalcWeightSum) {
    xllm::WeightMap weightQuant("/root/autodl-tmp/llama2_7b_chat_int8.bin");
    Data& data = weightQuant["model.layers.0.self_attn.q_proj.weight"];
    data.CalcWeightSum();
    printf("%i \n%i \n%i \n", data.weightSum[0], data.weightSum[1], data.weightSum[2]);
}

TEST(test_data, CalcWeightSum2) {
    
    int n = 2, m = 3;
    std::vector<uint8_t> uinput(n*m);
    for (int i = 0; i < n*m; i++) {
        uinput[i] = i+1;
    } 
    Data data(DataType::INT8, {n,m});
    data.Allocate();
    memcpy(data.cpuData, uinput.data(), n*m);
    data.CalcWeightSum();
    ASSERT_EQ(data.weightSum[0], 6);
    ASSERT_EQ(data.weightSum[1], 15);
}