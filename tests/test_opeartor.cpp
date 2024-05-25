#include <gtest/gtest.h>
#include "data.h"
#include "operator.h"
#include "param.h"

using namespace xllm;


TEST(test_operator, all) {
    WeightMap weight{"/root/autodl-tmp/llama2_7b_chat.bin"};

    Data tokens = Data(DataType::FLOAT32, {1, 5}, {1,2,3,4,5});
    Data hiddenStates(DataType::FLOAT32, {tokens.dims[1], 4096});
    Embedding(tokens, weight["embed_tokens.weight"], hiddenStates);
    ASSERT_EQ(hiddenStates.counts, 5*4096);
    ASSERT_NE(((float *)hiddenStates.cpuData)[0], 0);
    ASSERT_NE(((float *)hiddenStates.cpuData)[5*4096-1], 0);
    ASSERT_EQ(((float *)hiddenStates.cpuData)[5*4096], 0);

    Data attenInput(DataType::FLOAT32, hiddenStates.dims);
    RMSNorm(hiddenStates, weight["layers." + std::to_string(0) + ".input_layernorm.weight"],
        attenInput, 1e-6);

    std::string qWeightName = "layers." + std::to_string(0) + ".self_attn.q_proj.weight";
    std::string kWeightName = "layers." + std::to_string(0) + ".self_attn.k_proj.weight";
    std::string vWeightName = "layers." + std::to_string(0) + ".self_attn.v_proj.weight";
    Data q(DataType::FLOAT32, hiddenStates.dims), k(DataType::FLOAT32, hiddenStates.dims), v(DataType::FLOAT32, hiddenStates.dims);
    Linear(attenInput, weight[qWeightName], q);
    ASSERT_EQ(q.counts, 5*4096);

    ModelArgs params;
    int bsz = 1, seqlen = attenInput.dims[0];
    std::vector <int> qkvSize = {bsz, seqlen, params.num_attention_heads, -1};
    q.Reshape(qkvSize);

    std::vector<std::vector<float> > sin, cos;
    sin.resize(params.max_positions);
    cos.resize(params.max_positions);
    std::vector <float> invFreq;
    for (int i = 0; i < params.rotary_dim; i += 2) {
        invFreq.push_back(1.0 / pow(10000, (float)i / params.rotary_dim));
    }
    for (int i = 0; i < params.max_positions; i++) {
        sin[i].resize(params.rotary_dim);
        cos[i].resize(params.rotary_dim);
        for (int j = 0; j < invFreq.size(); j++) {
            sin[i][j] = ::sin((float)i * invFreq[j]);
            cos[i][j] = ::cos((float)i * invFreq[j]);
        }
    }
    std::vector <float> fsin, fcos;
    for (int i = 0; i < sin.size(); i++) {
        for (int j = 0; j < sin[0].size(); j++) {
            fsin.push_back(sin[i][j]);
            fcos.push_back(cos[i][j]);
        }
    }
    Data sinData, cosData;
    sinData.CopyFrom(Data(DataType::FLOAT32, {(int)sin.size(), (int)sin[0].size()}, fsin));
    cosData.CopyFrom(Data(DataType::FLOAT32, {(int)cos.size(), (int)cos[0].size()}, fcos));

    std::vector <float> vpids = std::vector <float> (seqlen, 0);
    for (int i = 0; i < seqlen; i++) {
        vpids[i] = i;
    }
    Data positionIds = Data(DataType::FLOAT32, {1, seqlen}, vpids);

    LlamaRotatePosition2D(q, positionIds, sinData, cosData, params.rotary_dim);
    ASSERT_EQ(q.counts, 4*4096);
}

TEST(test_operator, Linear_fp32) {
    Data input(DataType::FLOAT32, {2, 2}, {1,2,3,4});
    Data weight(DataType::FLOAT32, {2, 2}, {1,2,3,4});
    Data output(DataType::FLOAT32, {2,2});
    Linear(input, weight, output);
    ASSERT_EQ(((float*)output.cpuData)[0], 5);
    ASSERT_EQ(((float*)output.cpuData)[1], 11);
    ASSERT_EQ(((float*)output.cpuData)[2], 11);
    ASSERT_EQ(((float*)output.cpuData)[3], 25);
}


TEST(test_operator, PermuteSelf_axis_102) {
    std::vector<float> v;
    for (int i = 0; i < 24; i ++) {
        v.push_back(i);
    }
    Data input(DataType::FLOAT32, {2, 3, 4}, v);
    PermuteSelf(input, {1, 0, 2});
    ASSERT_EQ(((float*)input.cpuData)[0], 0);
    ASSERT_EQ(((float*)input.cpuData)[4], 12);
    ASSERT_EQ(((float*)input.cpuData)[8], 4);
    ASSERT_EQ(input.dims[0], 3);
    ASSERT_EQ(input.dims[1], 2);
    ASSERT_EQ(input.dims[2], 4);
    ASSERT_EQ(input.counts, 24);
}


TEST(test_operator, PermuteSelf_axis_0213) {
    std::vector<float> v;
    for (int i = 0; i < 48; i ++) {
        v.push_back(i);
    }
    Data input(DataType::FLOAT32, {2, 3, 4, 2}, v);
    PermuteSelf(input, {0, 2, 1, 3});
    ASSERT_EQ(((float*)input.cpuData)[0], 0);
    ASSERT_EQ(((float*)input.cpuData)[4], 16);
    ASSERT_EQ(((float*)input.cpuData)[6], 2);
    ASSERT_EQ(((float*)input.cpuData)[12], 4);
    ASSERT_EQ(((float*)input.cpuData)[24], 24);
    ASSERT_EQ(input.dims[0], 2);
    ASSERT_EQ(input.dims[1], 4);
    ASSERT_EQ(input.dims[2], 3);
    ASSERT_EQ(input.dims[3], 2);
    ASSERT_EQ(input.counts, 48);
}

TEST(test_operator, CatDirect) {
    // pastKey扩容
    Data pastKey(DataType::FLOAT32);
    std::vector<int> dims{2,10,3};
    pastKey.Expansion(dims);

    // 第一次concat
    std::vector<float> k_vector;
    for (int i = 0; i < 6; i ++) {
        k_vector.push_back(i);
    }
    Data k(DataType::FLOAT32, {2, 1, 3}, k_vector);
    CatDirect(pastKey, k, 1);
    // first head
    ASSERT_EQ(((float*)pastKey.cpuData)[0], 0);
    ASSERT_EQ(((float*)pastKey.cpuData)[1], 1);
    ASSERT_EQ(((float*)pastKey.cpuData)[2], 2);
    ASSERT_EQ(((float*)pastKey.cpuData)[3], 0);
    ASSERT_EQ(((float*)pastKey.cpuData)[4], 0);
    // second head
    ASSERT_EQ(((float*)pastKey.cpuData)[30], 3);
    ASSERT_EQ(((float*)pastKey.cpuData)[31], 4);
    ASSERT_EQ(((float*)pastKey.cpuData)[32], 5);
    ASSERT_EQ(((float*)pastKey.cpuData)[33], 0);

    ASSERT_EQ(pastKey.counts, 6);
    ASSERT_EQ(pastKey.expandCounts, 60);
    ASSERT_EQ(pastKey.assignBytes, 60*4);

    // 第二次concat
    std::vector<float> k_vector2;
    for (int i = 0; i < 6; i ++) {
        k_vector2.push_back(i*10);
    }
    Data k2(DataType::FLOAT32, {2, 1, 3}, k_vector2);
    CatDirect(pastKey, k2, 1);

    // first head
    ASSERT_EQ(((float*)pastKey.cpuData)[0], 0);
    ASSERT_EQ(((float*)pastKey.cpuData)[1], 1);
    ASSERT_EQ(((float*)pastKey.cpuData)[2], 2);
    ASSERT_EQ(((float*)pastKey.cpuData)[3], 0);
    ASSERT_EQ(((float*)pastKey.cpuData)[4], 10);
    ASSERT_EQ(((float*)pastKey.cpuData)[5], 20);
    ASSERT_EQ(((float*)pastKey.cpuData)[6], 0);
    // second head
    ASSERT_EQ(((float*)pastKey.cpuData)[30], 3);
    ASSERT_EQ(((float*)pastKey.cpuData)[31], 4);
    ASSERT_EQ(((float*)pastKey.cpuData)[32], 5);
    ASSERT_EQ(((float*)pastKey.cpuData)[33], 30);
    ASSERT_EQ(((float*)pastKey.cpuData)[34], 40);
    ASSERT_EQ(((float*)pastKey.cpuData)[35], 50);
    ASSERT_EQ(((float*)pastKey.cpuData)[36], 0);

    ASSERT_EQ(pastKey.counts, 12);
    ASSERT_EQ(pastKey.expandCounts, 60);
    ASSERT_EQ(pastKey.assignBytes, 60*4);
    ASSERT_EQ(pastKey.dims[0], 2);
    ASSERT_EQ(pastKey.dims[1], 2);
    ASSERT_EQ(pastKey.dims[2], 3);
}


TEST(test_operator, MatMulTransB){
    Data input0(DataType::FLOAT32, {2, 2, 1}, {1,2,3,4});
    Data output(DataType::FLOAT32, {2,2, 3});

    Data pastKey(DataType::FLOAT32);
    std::vector<int> dims{2,10,1};
    pastKey.Expansion(dims);
    std::vector<float> k_vector;
    for (int i = 0; i < 6; i ++) {
        k_vector.push_back(i+1);
    }
    Data k(DataType::FLOAT32, {2, 3, 1}, k_vector);
    CatDirect(pastKey, k, 1);

    MatMulTransB(input0, pastKey, output, 1);
    
    // head1
    ASSERT_EQ(((float*)output.cpuData)[0], 1);
    ASSERT_EQ(((float*)output.cpuData)[1], 2);
    ASSERT_EQ(((float*)output.cpuData)[2], 3);
    ASSERT_EQ(((float*)output.cpuData)[3], 2);
    ASSERT_EQ(((float*)output.cpuData)[4], 4);
    ASSERT_EQ(((float*)output.cpuData)[5], 6);

    // head2
    ASSERT_EQ(((float*)output.cpuData)[6], 12);
    ASSERT_EQ(((float*)output.cpuData)[7], 15);
    ASSERT_EQ(((float*)output.cpuData)[8], 18);
    ASSERT_EQ(((float*)output.cpuData)[9], 16);
    ASSERT_EQ(((float*)output.cpuData)[10], 20);
    ASSERT_EQ(((float*)output.cpuData)[11], 24);
}

TEST(test_operator, AttentionMask) {
    int seqLen = 3;
    std::vector<float> atten;
    for (int i = 0; i < 2*seqLen*3; i ++) {
        atten.push_back(i);
    } 
    Data attenWeights(DataType::FLOAT32, {1, 3,seqLen,3}, atten);
    std::vector <float> vmask = std::vector <float> (seqLen * seqLen, 0);   // mask matrix
    for (int i = 0; i < seqLen; i++) {
        for (int j = i + 1; j < seqLen; j++) {
            vmask[i * seqLen + j] = 1;   // mask标记为1
        }
    }
    Data attentionMask = Data(DataType::FLOAT32, {seqLen, seqLen}, vmask);
    int maskValue = -100000;
    AttentionMask(attenWeights, attentionMask, maskValue);

    ASSERT_NE(((float*)attenWeights.cpuData)[0], maskValue);
    ASSERT_EQ(((float*)attenWeights.cpuData)[1], maskValue);
    ASSERT_EQ(((float*)attenWeights.cpuData)[2], maskValue);

    ASSERT_NE(((float*)attenWeights.cpuData)[3], maskValue);
    ASSERT_NE(((float*)attenWeights.cpuData)[4], maskValue);
    ASSERT_EQ(((float*)attenWeights.cpuData)[5], maskValue);

    ASSERT_NE(((float*)attenWeights.cpuData)[6], maskValue);
    ASSERT_NE(((float*)attenWeights.cpuData)[7], maskValue);
    ASSERT_NE(((float*)attenWeights.cpuData)[8], maskValue);
}


TEST(test_operator, SoftMax) {
    Data input(DataType::FLOAT32, {2,3}, {0.1,0.2,-10000,3,4, -10000});
    SoftMax(input, input, -1);
    ASSERT_EQ((((float*)input.cpuData)[0] + ((float*)input.cpuData)[1]), 1);
    ASSERT_EQ(((float*)input.cpuData)[2], 0);
    ASSERT_EQ((((float*)input.cpuData)[3] + ((float*)input.cpuData)[4]), 1);
    ASSERT_EQ(((float*)input.cpuData)[5], 0);
}

// 这个测试需要关闭avx
TEST(test_operator, MultiplyMultiThread) {
    
    int n = 2, m = 3, k = 4;
    std::vector<uint8_t> uinput(n*m);
    for (int i = 0; i < n*m; i++) {
        uinput[i] = i+1;
    } 
    std::vector<uint8_t> weight{1, 5,9, 2,6,10,3,7,11,4,8,12};
    Data output(DataType::FLOAT32, {n,k});
    output.Allocate();
    MultiplyMultiThread(uinput.data(), weight.data(), (int32_t *)output.cpuData, n, m, k, 1);
    ASSERT_EQ(((int32_t*)output.cpuData)[0], 38);
    ASSERT_EQ(((int32_t*)output.cpuData)[1], 44);
    ASSERT_EQ(((int32_t*)output.cpuData)[2], 50);
    ASSERT_EQ(((int32_t*)output.cpuData)[3], 56);
    ASSERT_EQ(((int32_t*)output.cpuData)[4], 83);
    ASSERT_EQ(((int32_t*)output.cpuData)[5], 98);
    ASSERT_EQ(((int32_t*)output.cpuData)[6], 113);
    ASSERT_EQ(((int32_t*)output.cpuData)[7], 128);
}

TEST(test_operator, MatMul) {
    
}

TEST(test_operator, AddTo) {

}

TEST(test_operator, MulTo) {

}

TEST(test_operator, Silu) {

}