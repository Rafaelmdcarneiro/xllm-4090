#include "model.h"
#include "data.h"

namespace xllm {

    LlamaModel::LlamaModel(const std::string &weightPath, const std::string &tokenPath): 
        weight(weightPath), tokenizer(tokenPath) {      
        
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
        sinData.CopyFrom(Data(DataType::FLOAT32, {(int)sin.size(), (int)sin[0].size()}, fsin));
        cosData.CopyFrom(Data(DataType::FLOAT32, {(int)cos.size(), (int)cos[0].size()}, fcos));
        
        deviceMap = GetDeviceMap();
        auto st = std::chrono::system_clock::now();
        WarmUp();
        printf("Warm up spend %f s.\n", xllm::GetSpan(st, std::chrono::system_clock::now()));
    }

    std::vector<float> LlamaModel::MakeInput(std::vector<float> &history, int round, const std::string &input) {
        std::string input_trim = trim(input);
        std::string prompt = (round == 0 ? pre_prompt : "") + B_INST + input_trim + E_INST;
        std::vector<float> inputIds = tokenizer.Encode(prompt, true);
        history.insert(history.end(), inputIds.begin(), inputIds.end());
        return history;
    }

    void LlamaModel::MakeHistory(std::vector<float> &history, int round, const std::string &input, const std::string &output) {
        std::string input_trim = trim(input);
        std::string output_trim = trim(output);
        std::string last =  B_INST + input_trim + E_INST + output_trim;
        std::vector<float> lastchat = tokenizer.Encode(last, true ,true);
        history.insert(history.end(), lastchat.begin(), lastchat.end());
    }

    void LlamaModel::WarmUp() {
        printf("Warmup...\n");
        Data inputIds = Data(DataType::FLOAT32, {1, 1}, {1});
        Data attentionMask = Data(DataType::FLOAT32, {1, 1}, {0});
        Data positionIds = Data(DataType::FLOAT32, {1, 1}, {0, 0});

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < params.block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(DataType::FLOAT32),
                                                   Data(DataType::FLOAT32)));
        }
        Forward(inputIds, attentionMask, positionIds, pastKeyValues);
        printf("finish.\n");
    }

    std::string LlamaModel::Response(const std::vector<float>& input, RuntimeResult retCb,
                                     const GenerationConfig &generationConfig) {

        auto st = std::chrono::system_clock::now();

        std::vector <float> ids;
        int seqLen = input.size();
        Data inputIds(DataType::FLOAT32, {1, seqLen}, input);

        std::vector <float> vmask = std::vector <float> (seqLen * seqLen, 0);   // mask matrix
        std::vector <float> vpids = std::vector <float> (seqLen, 0);
        for (int i = 0; i < seqLen; i++) {
            vpids[i] = i;
            for (int j = i + 1; j < seqLen; j++) {
                vmask[i * seqLen + j] = 1;   // mask标记为1
            }
        }
        Data attentionMask = Data(DataType::FLOAT32, {seqLen, seqLen}, vmask);
        Data positionIds = Data(DataType::FLOAT32, {1, seqLen}, vpids);

        std::vector <std::pair <Data, Data> > pastKeyValues;  // KV cache
        for (int i = 0; i < params.block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(DataType::FLOAT32), Data(DataType::FLOAT32)));
        }

        std::string retString = "";
        int len = seqLen;
        std::vector <float> results;
        int index = 0;

        LastTokensManager tokens (1, generationConfig.last_n);
        while (true) {
            // auto st = std::chrono::system_clock::now();

            int ret = Forward(inputIds, attentionMask, positionIds, pastKeyValues, generationConfig, tokens);
            tokens.units[0].Push(ret);
            if (ret == tokenizer.eos_id) {
                break;
            }

            results.push_back(ret);
            std::string curString = tokenizer.Decode(Data(DataType::FLOAT32, {(int)results.size()}, results));
            retString += curString;
            if (retCb)
                retCb(index, curString.c_str());
            index++;

            if (index == generationConfig.output_token_limit) {
                break;
            }
            results.clear();
            
            attentionMask.ToDevice(DataDevice::CPU);
            positionIds.ToDevice(DataDevice::CPU);
            inputIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, {(float)ret}));
            attentionMask = Data();
            positionIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, {(float)len}));
            len++;
            if (index == generationConfig.output_token_limit) {
                break;
            }

            // printf("spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
        }
        if (retCb)
            retCb(-1, retString.c_str());

        return retString;
    }

    void LlamaModel::ResponseBatch(const std::vector<std::vector<float>> &prompts, std::vector<std::string> &outputs,
                                   RuntimeResultBatch retCb, const GenerationConfig &generationConfig) {
        int batch = prompts.size();
        outputs.clear();
        outputs.resize(batch, "");

        std::vector<Data> inputTokens;
        std::vector<int> seqLens;
        seqLens.resize(batch);
        int maxLen = 0;
        for (int i = 0; i < batch; i++) {
            seqLens[i] = prompts[i].size();
            maxLen = std::max(maxLen, seqLens[i]);
            inputTokens.push_back(Data(DataType::FLOAT32, {1, seqLens[i]} , prompts[i]));
        }

        std::vector <float> ids = std::vector <float> (batch * maxLen, 0);
        std::vector <float> vpids = std::vector <float> (batch * maxLen, 0);
        std::vector <float> vmask = std::vector <float> (batch * maxLen * maxLen, 0);
        for (int i = 0; i < batch; i++) {
            int len = seqLens[i], base = maxLen - len;
            for (int j = 0; j < len; j++) {
                ids[i * maxLen + base + j] = ((float*)inputTokens[i].cpuData)[j];
                vpids[i * maxLen + base + j] = j;
            }

            std::fill(vmask.data() + i * maxLen * maxLen,
                      vmask.data() + i * maxLen * maxLen + (maxLen - len) * maxLen, 1.0);
            for (int j = maxLen - len; j < maxLen; j++) {
                std::fill(vmask.data() + i * maxLen * maxLen + j * maxLen,
                          vmask.data() + i * maxLen * maxLen + j * maxLen + maxLen - len, 1.0);
            }
            for (int j = 0; j < len; j++) {
                for (int k = j + 1; k < len; k++) {
                    vmask[i * maxLen * maxLen + (base + j) * maxLen + base + k] = 1;
                }
            }
        }

        Data inputIds = Data(DataType::FLOAT32, {batch, maxLen}, ids);
        Data attentionMask = Data(DataType::FLOAT32, {batch, maxLen, maxLen}, vmask);
        Data positionIds = Data(DataType::FLOAT32, {batch, maxLen}, vpids);

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < params.block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(DataType::FLOAT16),
                                                   Data(DataType::FLOAT16)));
        }

        std::string retString = "";
        int index = 0;  // 用来区分prefill和decode

        LastTokensManager tokensManager (batch, generationConfig.last_n);
        while (true) {
            // auto st = std::chrono::system_clock::now();
            std::vector <int> ret = ForwardBatch(batch, inputIds, attentionMask, positionIds, pastKeyValues,
                                                 generationConfig, tokensManager);
            // printf("ForwardBatch spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));

            std::vector <float> fret;
            std::vector <float> results;
            std::vector <std::string> curStrings;
            std::vector<int> removedBatch;
            int origin_batch = batch;
            for (int i = 0; i < batch; i++) {

                if (ret[i] == tokenizer.eos_id) {
                    seqLens.erase(seqLens.begin() + i);
                    ret.erase(ret.begin() + i);
                    printf("[ model output: \"%s\"]\n", outputs[i].c_str());
                    outputs.erase(outputs.begin() + i);
                    removedBatch.push_back(i + removedBatch.size());
                    batch--;
                    i--;
                    continue;
                }

                fret.push_back(ret[i]);
                
                results.push_back(ret[i]);
                std::string curString = tokenizer.Decode(Data(DataType::FLOAT32, {(int)results.size()}, results));
                outputs[i] += curString;
                curStrings.push_back(curString);
                results.clear();
            }

            if (retCb) 
                retCb(index, curStrings);

            if (batch == 0) {
                break;
            }

            if (removedBatch.size()>0) {
                // auto st = std::chrono::system_clock::now();
                // #pragma omp parallel for
                for (int block = 0; block < params.block_cnt; block++) {
                    pastKeyValues[block].first.removeBatch(removedBatch, origin_batch);
                    pastKeyValues[block].second.removeBatch(removedBatch, origin_batch);
                }
                // printf("removeBatch spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
            }
            
            for (int i = 0; i < batch; i++) {
                tokensManager.units[i].Push(ret[i]);
            }

            index++;
            maxLen++;
            
            std::vector <float> pids = std::vector <float> (batch);
            std::vector <float> vmasks = std::vector <float> (batch * maxLen, 0.0f);
            for (int i = 0; i < batch; i++) {
                pids[i] = seqLens[i];
                seqLens[i]++;
                for (int j = 0; j < maxLen - seqLens[i]; j++) {
                    vmasks[i * maxLen + j] = 1.0f;
                }
            }

            attentionMask.ToDevice(DataDevice::CPU);
            positionIds.ToDevice(DataDevice::CPU);
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {batch, 1, maxLen}, vmasks));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {batch, 1}, pids));
            inputIds.CopyFrom(Data(DataType::FLOAT32, {batch, 1}, fret));

            if (index == generationConfig.output_token_limit) {
                break;
            }

        }
        if (retCb)
            retCb(-1, outputs);
    }

    int LlamaModel::Forward(const Data &inputIds, const Data &attentionMask, const Data &positionIds, 
                            std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <float> *retLogits) {
        
        int bsz = 1, seqlen = inputIds.dims[1];
        Data hiddenStates(DataType::FLOAT32, {bsz, seqlen, params.embed_dim});
        Data attenInput(DataType::FLOAT32, hiddenStates.dims);
        Data q(DataType::FLOAT32, hiddenStates.dims), k(DataType::FLOAT32, hiddenStates.dims), v(DataType::FLOAT32, hiddenStates.dims);
        Data attenOutput(DataType::FLOAT32, {bsz, params.num_attention_heads, seqlen, params.hidden_size/params.num_attention_heads});
        Data attenLastOutput(DataType::FLOAT32, {bsz, seqlen, params.hidden_size});
        Data w1(DataType::FLOAT32, {bsz, seqlen, params.intermediate_size});
        Data w3(DataType::FLOAT32, {bsz, seqlen, params.intermediate_size});
        Data w2(DataType::FLOAT32, {bsz, seqlen, params.hidden_size});

        Embedding(inputIds, weight["model.embed_tokens.weight"], hiddenStates);
        for (int i = 0; i < params.block_cnt; i++) {
            // ApplyDeviceMap(this->deviceMap, i + 1, params.block_cnt);
            RMSNorm(hiddenStates, weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"],
                    attenInput, 1e-6);
            // attenInput.ToDevice(DataDevice::CPU);
            std::string qWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            std::string kWeightName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.weight";
            std::string vWeightName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";

            // 1.1 Get q, k, v
            q.Reshape(hiddenStates.dims);
            k.Reshape(hiddenStates.dims);
            v.Reshape(hiddenStates.dims);
            Linear(attenInput, weight[qWeightName], q);
            // q.ToDevice(DataDevice::CPU);
            Linear(attenInput, weight[kWeightName], k);
            // k.ToDevice(DataDevice::CPU);
            Linear(attenInput, weight[vWeightName], v);
            // v.ToDevice(DataDevice::CPU);

            std::vector <int> qkvSize = {bsz, seqlen, params.num_attention_heads, -1};
            q.Reshape(qkvSize);
            k.Reshape(qkvSize);

            LlamaRotatePosition2D(q, positionIds, sinData, cosData, params.rotary_dim);
            LlamaRotatePosition2D(k, positionIds, sinData, cosData, params.rotary_dim);
            // q.ToDevice(DataDevice::CPU);
            // k.ToDevice(DataDevice::CPU);

            qkvSize = {bsz * seqlen, params.num_attention_heads, -1};
            q.Reshape(qkvSize);
            k.Reshape(qkvSize);
            v.Reshape(qkvSize);

            std::vector<int> axisData = {1, 0, 2};
            PermuteSelf(q, axisData);
            // q.ToDevice(DataDevice::CPU);
            PermuteSelf(k, axisData);
            // k.ToDevice(DataDevice::CPU);
            PermuteSelf(v, axisData);
            // v.ToDevice(DataDevice::CPU);

            Data &pastKey = pastKeyValues[i].first, &pastValue = pastKeyValues[i].second;
            int unitLen = 64;   // 每次扩容的seq_len是unitLen的倍数
#ifdef USE_CUDA
                unitLen = 128;
#endif
            while ((pastKey.dims.size() == 0 && (pastKey.expandDims.size() == 0 || k.dims[1] > pastKey.expandDims[1]))
                   || (pastKey.dims.size() > 0 && pastKey.dims[1] + k.dims[1] > pastKey.expandDims[1])) {
                std::vector <int> newDims;
                if (pastKey.counts == 0 || pastKey.dims.size() == 0) {
                    newDims = std::vector <int> {k.dims[0], ((k.dims[1] - 1) / unitLen + 1) * unitLen, k.dims[2]};
                } else {
                    newDims = pastKey.dims;
                    newDims[1] += ((k.dims[1] - 1) / unitLen + 1) * unitLen;
                }
                pastKey.Expansion(newDims);
            }
            while ((pastValue.dims.size() == 0 && (pastValue.expandDims.size() == 0 || v.dims[1] > pastValue.expandDims[1]))
                   || (pastValue.dims.size() > 0 && pastValue.dims[1] + v.dims[1] > pastValue.expandDims[1])) {
                std::vector <int> newDims;
                if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                    newDims = std::vector <int> {v.dims[0], ((v.dims[1] - 1) / unitLen + 1) * unitLen, v.dims[2]};
                } else {
                    newDims = pastValue.dims;
                    newDims[1] += ((v.dims[1] - 1) / unitLen + 1) * unitLen;
                }
                pastValue.Expansion(newDims);
            }

            CatDirect(pastKey, k, 1);
            // pastKey.ToDevice(DataDevice::CPU);
            CatDirect(pastValue, v, 1);
            // pastValue.ToDevice(DataDevice::CPU);
            
            // 1.2 Attention
            // 1.2.0 q * k^T
            // q: {num_attention_heads, bsz * seqlen, hidden_size/num_attention_heads}
            // pastKey: {num_attention_heads, k_seqlen, hidden_size/num_attention_heads} 不同head之间的内存不连续
            Data attenWeights(DataType::FLOAT32, {q.dims[0], q.dims[1], pastKey.dims[1]});
            MatMulTransB(q, pastKey, attenWeights, 1.0 / sqrt(params.head_dim));
            // attenWeights.ToDevice(DataDevice::CPU);
            attenWeights.Reshape({1, attenWeights.dims[0], attenWeights.dims[1], attenWeights.dims[2]});
            if (attentionMask.dims.size() != 0) {
                AttentionMask(attenWeights, attentionMask, -10000);
                // attenWeights.ToDevice(DataDevice::CPU);
            }

            SoftMax(attenWeights, attenWeights, -1);
            // attenWeights.ToDevice(DataDevice::CPU);
            // attenWeights: {1, num_attention_heads, bsz * seqlen, k_seqlen}
            // pastValue: {num_attention_heads, k_seqlen, hidden_size/num_attention_heads} 不同head之间的内存不连续
            // attenOutput: {1, num_attention_heads, bsz * seqlen, hidden_size/num_attention_heads}
            attenOutput.Reshape({1, params.num_attention_heads, bsz * seqlen, -1});
            MatMul(attenWeights, pastValue, attenOutput);
            // attenOutput.ToDevice(DataDevice::CPU);
            
            attenOutput.Reshape({attenOutput.dims[1], attenOutput.dims[2], attenOutput.dims[3]});
            PermuteSelf(attenOutput, axisData);
            // {bsz, seqLen, hidden_size}
            attenOutput.Reshape({bsz, seqlen, -1});
            // attenOutput.ToDevice(DataDevice::CPU);
            
            // weight[oWeightName]: {hidden_size, hidden_size}
            Linear(attenOutput, weight[oWeightName], attenLastOutput);
            // attenLastOutput.ToDevice(DataDevice::CPU);
            AddTo(hiddenStates, attenLastOutput);
            // hiddenStates.ToDevice(DataDevice::CPU);

            // 2. mlp
            RMSNorm(hiddenStates, weight["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"], attenInput, 1e-6);
            // attenInput.ToDevice(DataDevice::CPU);
            Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"],  w1);
            // w1.ToDevice(DataDevice::CPU);
            Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.up_proj.weight"],  w3);
            // w3.ToDevice(DataDevice::CPU);
            Silu(w1, w1);
            // w1.ToDevice(DataDevice::CPU);
            MulTo(w1, w3);
            // w1.ToDevice(DataDevice::CPU);
            Linear(w1, weight["model.layers." + std::to_string(i) + ".mlp.down_proj.weight"], w2);
            // w2.ToDevice(DataDevice::CPU);
            AddTo(hiddenStates, w2);
            // hiddenStates.ToDevice(DataDevice::CPU);
        }

        RMSNorm(hiddenStates, weight["model.norm.weight"], hiddenStates, 1e-6);
        // hiddenStates.ToDevice(DataDevice::CPU);
        Data logits(DataType::FLOAT32, {bsz, hiddenStates.dims[1], params.vocab_size});
        Linear(hiddenStates, weight["lm_head.weight"], logits);
        logits.ToDevice(DataDevice::CPU);
        
        // 采样
        int lastRet = -1;
        int base = logits.dims[1] - 1;

        if (generationConfig.IsSimpleGreedy()) {
            std::pair <float, int> ret = std::make_pair(-1e9, -1);
            for (int i = 0; i < logits.dims.back(); i++) {
                ret = max(ret, std::make_pair(((float*)logits.cpuData)[base * logits.dims.back() + i], i));
            }
            lastRet = ret.second;
        } else {
            // std::vector<float> ret;
            // for (int i = 0; i < params.vocab_size; i++) {
            //     ret.push_back(((float*)logits.cpuData)[base * params.vocab_size + i]/generationConfig.temperature);
            // }
            // Data input(DataType::FLOAT32, {1, params.vocab_size}, ret);
            // Data output(DataType::FLOAT32, {1, params.vocab_size});
            // SoftMax(input, output, -1);
            // lastRet = sample_top_p(output.cpuData, generationConfig);
            lastRet = TOPKSampling(logits, base - 1, generationConfig, lastTokens.units[0]);
        }

        return lastRet; 
    }

    std::vector <int> LlamaModel::ForwardBatch(int batch, const xllm::Data &inputIds, const xllm::Data &attentionMask,
                            const xllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <std::vector <float>*> *retLogits) {
        int bsz = batch, seqlen = inputIds.dims[1];
        Data hiddenStates(DataType::FLOAT32, {bsz, seqlen, params.embed_dim});
        Data attenInput(DataType::FLOAT32, hiddenStates.dims);
        Data q(DataType::FLOAT32, hiddenStates.dims), k(DataType::FLOAT32, hiddenStates.dims), v(DataType::FLOAT32, hiddenStates.dims);
        Data attenOutput(DataType::FLOAT32, {bsz, params.num_attention_heads, seqlen, params.hidden_size/params.num_attention_heads});
        Data attenLastOutput(DataType::FLOAT32, {bsz, seqlen, params.hidden_size});
        Data w1(DataType::FLOAT32, {bsz, seqlen, params.intermediate_size});
        Data w3(DataType::FLOAT32, {bsz, seqlen, params.intermediate_size});
        Data w2(DataType::FLOAT32, {bsz, seqlen, params.hidden_size});

        // inputIds: (bsz, seqlen)
        // hiddenStates: (bsz, seqlen, embed_dim)
        Embedding(inputIds, this->weight["model.embed_tokens.weight"], hiddenStates);
        for (int i = 0; i < params.block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, params.block_cnt);
            // attenInput: (bsz, seqlen, embed_dim)
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"],
                    attenInput, 1e-6);
            std::string qWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            std::string kWeightName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.weight";
            std::string vWeightName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";

            // 1.1 Get q, k, v
            q.Reshape(hiddenStates.dims);
            k.Reshape(hiddenStates.dims);
            v.Reshape(hiddenStates.dims);
            // q/k/v: (bsz, seqlen, embed_dim)
            Linear(attenInput, weight[qWeightName], q);
            Linear(attenInput, weight[kWeightName], k);
            Linear(attenInput, weight[vWeightName], v);

            std::vector <int> qkvSize = {bsz, seqlen, params.num_attention_heads, -1};
            q.Reshape(qkvSize);
            k.Reshape(qkvSize);
            v.Reshape(qkvSize);

            // q/k: (bsz, seqlen, params.num_attention_heads, embed_dim/params.num_attention_heads)
            LlamaRotatePosition2D(q, positionIds, sinData, cosData, params.rotary_dim);
            LlamaRotatePosition2D(k, positionIds, sinData, cosData, params.rotary_dim);

            // q/k/v -> (bsz, params.num_attention_heads, seqlen, embed_dim/params.num_attention_heads)
            PermuteSelf(q, {0, 2, 1, 3});
            PermuteSelf(k, {0, 2, 1, 3});
            PermuteSelf(v, {0, 2, 1, 3});

            // q/k/v: (bsz*params.num_attention_heads, seqlen, embed_dim/params.num_attention_heads)
            qkvSize = {bsz * params.num_attention_heads, seqlen, -1};
            q.Reshape(qkvSize);
            k.Reshape(qkvSize);
            v.Reshape(qkvSize);

            Data &pastKey = pastKeyValues[i].first, &pastValue = pastKeyValues[i].second;
            int unitLen = 64;
#ifdef USE_CUDA
                unitLen = 64;
#endif
            while ((pastKey.dims.size() == 0 && (pastKey.expandDims.size() == 0 || k.dims[1] > pastKey.expandDims[1]))
                   || (pastKey.dims.size() > 0 && pastKey.dims[1] + k.dims[1] > pastKey.expandDims[1])) {
                std::vector <int> newDims;
                if (pastKey.Count(0) == 0 || pastKey.dims.size() == 0) {
                    newDims = std::vector <int> {k.dims[0], ((k.dims[1] - 1) / unitLen + 1) * unitLen, k.dims[2]};
                } else {
                    newDims = pastKey.dims;
                    newDims[1] += ((k.dims[1] - 1) / unitLen + 1) * unitLen;
                }
                pastKey.Expansion(newDims);
            }
            while ((pastValue.dims.size() == 0 && (pastValue.expandDims.size() == 0 || v.dims[1] > pastValue.expandDims[1]))
                   || (pastValue.dims.size() > 0 && pastValue.dims[1] + v.dims[1] > pastValue.expandDims[1])) {
                std::vector <int> newDims;
                if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                    newDims = std::vector <int> {v.dims[0], ((v.dims[1] - 1) / unitLen + 1) * unitLen, v.dims[2]};
                } else {
                    newDims = pastValue.dims;
                    newDims[1] += ((v.dims[1] - 1) / unitLen + 1) * unitLen;
                }
                pastValue.Expansion(newDims);
            }

            // struct FP16ToFP32Manager {
            //     float dict[65536];  // 2^16

            //     FP16ToFP32Manager() {
            //         for (uint16_t i = 0; i < 65535; i++) {
            //             dict[i] = half_to_float(i);
            //         }
            //     }
            // } fp16tofp32;
            
            // k.ToDevice(DataDevice::CPU);
            // pastKey.ToDevice(DataDevice::CPU);
            // uint16_t *pastKeyData = (uint16_t *) pastKey.cpuData;
            // pastKey: (bsz * params.num_attention_heads, historyLen, embed_dim/params.num_attention_heads)
            CatDirectFP16(pastKey, k, 1);
            // pastKey.ToDevice(DataDevice::CPU);
            // pastKeyData = (uint16_t *) pastKey.cpuData;
            CatDirectFP16(pastValue, v, 1);

            // attenOutput.Reshape({bsz*params.num_attention_heads, seqlen, -1});
            // Attention(q, pastKey, pastValue, attentionMask, attenOutput, 
            // q.dims[0] / pastKey.dims[0], 1.0 / sqrt(params.head_dim), 1);

            // 1.2 Attention
            // 1.2.0 Attention score: q * k^T
            // attenWeights: (bsz * params.num_attention_heads, seqlen, historyLen)
            Data attenWeights(DataType::FLOAT32, {q.dims[0], q.dims[1], pastKey.dims[1]});
            MatMulTransBFP16(q, pastKey, attenWeights, 1.0 / sqrt(params.head_dim));
            // attenWeights.ToDevice(DataDevice::CPU);
            // pastKey.ToDevice(DataDevice::CPU);
            // pastKeyData = (uint16_t *) pastKey.cpuData;

            attenWeights.Reshape({1, attenWeights.dims[0], attenWeights.dims[1], attenWeights.dims[2]});
            if (attentionMask.dims.size() != 0) {
                AttentionMask(attenWeights, attentionMask, -10000);
            }
            SoftMax(attenWeights, attenWeights, -1);
            attenOutput.Reshape({bsz*params.num_attention_heads, seqlen, -1});
            // attenOutput: (bsz*params.num_attention_heads, seqlen, embed_dim/params.num_attention_heads)
            MatMulFP16(attenWeights, pastValue, attenOutput);

            PermuteSelf(attenOutput, {1, 0, 2});
            attenOutput.Reshape({seqlen, bsz, -1});
            // attenOutput: (bsz, seqlen, embed_dim)
            PermuteSelf(attenOutput, {1, 0, 2});

            // attenLastOutput: (bsz, seqlen, embed_dim)
            Linear(attenOutput, weight[oWeightName], attenLastOutput);
            AddTo(hiddenStates, attenLastOutput);
            // 2. mlp
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"], attenInput, 1e-6);
            Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"], w1);
            Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.up_proj.weight"], w3);
            Silu(w1, w1);
            MulTo(w1, w3);
            Linear(w1, weight["model.layers." + std::to_string(i) + ".mlp.down_proj.weight"], w2);
            AddTo(hiddenStates, w2);
        }

        Data tempHiddenStates(DataType::FLOAT32, {bsz, 1, params.embed_dim});
        Data *lastHiddenStates;
        if (seqlen > 1) {
            Split(hiddenStates, 1, seqlen - 1, seqlen, tempHiddenStates);
            lastHiddenStates = &tempHiddenStates;
        } else {
            lastHiddenStates = &hiddenStates;
        }

        std::vector <int> lastRet;
        {
            auto &hiddenStates = *lastHiddenStates;
            RMSNorm(hiddenStates, weight["model.norm.weight"], hiddenStates, 1e-6);
            Data logits(DataType::FLOAT32, {bsz, hiddenStates.dims[1], params.vocab_size});
            Linear(hiddenStates, weight["lm_head.weight"], logits);
            Data topk(DataType::FLOAT32, {bsz, 1, 2});
            if (generationConfig.IsSimpleGreedy()) {
                TopK(logits, topk, 1);
                topk.ToDevice(DataDevice::CPU);
                for (int b = 0; b < batch; b++) {
                    int base = b;
                    lastRet.push_back((int) (((float *) topk.cpuData)[base * 2] + 1e-3));
                }
            } else {
                // for (int b = 0; b < batch; b++) {
                //     int base = b * logits.dims[1] + logits.dims[1] - 1;
                //     lastRet.push_back(LLMSampling(logits, base, generationConfig, lastTokens.units[b]));
                // }
            }
        }

        return lastRet;
    }

    struct Random {
        Random () {
            srand(time(NULL));
        }

        float randP() {
            return (float)(rand() % 10001) * 0.0001;
        }
    };
    
    Random fastllmRandom;

    int LlamaModel::TOPKSampling(Data &logits, int outerOffset,
                    const GenerationConfig &config, const LastTokensUnit &tokens) {
        int vocabSize = logits.dims.back();
        float *base = ((float*)logits.cpuData) + outerOffset * vocabSize;

        // 降低重复token的输出概率
        if (fabs(config.repeat_penalty - 1.0) > 1e-6) {
            for (int id : tokens.tokenSet) {
                base[id] = (base[id] < 0 ? base[id] * config.repeat_penalty : base[id] / config.repeat_penalty);
            }
        }

        float invTemp = 1.0f / config.temperature;
        std::vector <std::pair <float, int> > v;
        for (int i = 0; i < vocabSize; i++) {
            v.push_back(std::make_pair(-base[i] * invTemp, i));
        }
        int topk = std::min(vocabSize, config.top_k);
        std::partial_sort(v.begin(), v.begin() + topk, v.end());
        float psum = 0.0, maxValue = -v.begin()->first;
        std::vector <float> ps;
        for (int i = 0; i < topk; i++) {
            ps.push_back(expf(-v[i].first - maxValue));
            psum += ps.back();
        }
        float curSum = 0.0;
        for (int i = 0; i < topk; i++) {
            ps[i] /= psum;
            curSum += ps[i];
            if (curSum > config.top_p) {
                topk = i + 1;
                break;
            }
        }
        float rnd = fastllmRandom.randP() * curSum;
        curSum = 0.0;
        for (int i = 0; i < topk; i++) {
            curSum += ps[i];
            if (curSum > rnd || i == topk - 1) {
                return v[i].second;
            }
        }
        return -1;
    }

    int LlamaModel::compare_indexed_float(const void* a, const void* b) {
        IndexedFloat* indexed_a = (IndexedFloat*)a;
        IndexedFloat* indexed_b = (IndexedFloat*)b;
        return (indexed_b->value > indexed_a->value) ? 1 : ((indexed_b->value < indexed_a->value) ? -1 : 0);
    }

    int LlamaModel::sample_top_p(uint8_t* probs, const GenerationConfig& generationConfig) {
        int n = params.vocab_size;
        IndexedFloat probs_sort[n];
        for (int i = 0; i < n; i++) {
            probs_sort[i].value = ((float*)probs)[i];
            probs_sort[i].index = i;
        }
        qsort(probs_sort, n, sizeof(IndexedFloat), compare_indexed_float);

        float accum = 0.f;
        int p = 0;
        for (; accum<=generationConfig.top_p && p < n; p++) {
            accum += probs_sort[p].value;
        }

        // random float32 in [0,1)
        std::random_device rd; // 用于生成随机种子的设备
        std::mt19937 gen(rd()); // Mersenne Twister 伪随机数生成器
        std::uniform_real_distribution<float> dis(0.0f, 1.0f); // 均匀分布在[0, 1)之间的浮点数

        float r = dis(gen) * accum;
        float cdf = 0.0f;
        for (int i = 0; i < p; i++) {
            cdf += probs_sort[i].value;
            if (r < cdf) {
                return probs_sort[i].index;
            }
        }
        return probs_sort[p-1].index; // in case of rounding errors
    }
}