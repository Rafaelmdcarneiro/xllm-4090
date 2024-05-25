#include "param.h"
#include "utils.h"
#include "file.h"

namespace xllm{
    Tokenizer::Tokenizer(const std::string path){
        FileReader reader(path);
        vocab_size = reader.ReadInt();
        bos_id = reader.ReadInt();
        eos_id = reader.ReadInt();

        float score;
        std::string token;
        for (int i = 0; i < vocab_size; i++) {
            score = reader.ReadFloat();
            token = reader.ReadString();
            token_id[token] = i;
            id_score.push_back(score);
            id_token.push_back(token);
        }
    }
    
    std::vector<float> Tokenizer::Encode(const std::string &s, bool bos, bool eos) {

        std::vector<int> tokens;
        // first encode every individual byte in the input string
        for (char c: s) 
            tokens.push_back(token_id[std::string{c}]);

        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        int n_tokens = s.length(); // the number of tokens
        while (1) {
            float best_score = -1e10;
            int best_id = -1;
            int best_idx = -1;

            std::string merString;
            for (int i=0; i < n_tokens-1; i++) {
                merString = id_token[tokens[i]] + id_token[tokens[i+1]];
                if (token_id.find(merString) != token_id.end()){
                    int id = token_id[merString];
                    if (id_score[id] > best_score) {
                        // this merge pair exists in vocab! record its score and position
                        best_score = id_score[id];
                        best_id = id;
                        best_idx = i;
                    }
                }
            }

            if (best_idx == -1)
                break; // we couldn't find any more pairs to merge, so we're done

            // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
            tokens[best_idx] = best_id;
            // delete token at position best_idx+1, shift the entire sequence back 1
            for (int i = best_idx+1; i < n_tokens-1; i++) {
                tokens[i] = tokens[i+1];
            }
            n_tokens--; // token length decreased
        }

        std::vector<float> tokens_output;
        if(bos)
            tokens_output.push_back(bos_id);
        for(int i=0; i<n_tokens; i++)
            tokens_output.push_back(tokens[i]);
        if(eos)
            tokens_output.push_back(eos_id);
        
        return tokens_output;
    }
        

    std::string Tokenizer::Decode(const Data& data) {
        return id_token[((float *) data.cpuData)[0]];
    }

    WeightMap::WeightMap(const std::string &fileName){
        FileReader buffer(fileName);

        int keyValueLen = buffer.ReadInt();
        for (int i = 0; i < keyValueLen; i++) {
            std::string key = buffer.ReadString();
            std::string value = buffer.ReadString();
            params[key] = value;
        }

        // for(auto& pair:params)
        //     std::cout << "Key: " << pair.first << " Value: " << pair.second << std::endl;
    
        int weightLen = buffer.ReadInt();
        for (int i = 0; i < weightLen; i++) {
            std::string name = buffer.ReadString();
            //printf("%s\n", name.c_str());
            int dimsSize = buffer.ReadInt();
            //printf("size = %d\n", dimsSize);
            std::vector <int> dims;
            for (int j = 0; j < dimsSize; j++) {
                int x = buffer.ReadInt();
                dims.push_back(x);
            }
            DataType dataType = (DataType)buffer.ReadInt();
            weight[name] = Data(dataType, dims);
            weight[name].Allocate();
            if (dataType == DataType::FLOAT32 || dataType == DataType::BFLOAT16 || dataType == DataType::FLOAT16) {
                buffer.ReadBytes(weight[name].cpuData, weight[name].bytes);
            } else if (dataType == DataType::INT8 || dataType == DataType::INT4) {
		            int bit = (dataType == DataType::INT4 ? 4 : 8);
		            weight[name].perChannelAxis = buffer.ReadInt();
		            int k = weight[name].perChannelAxis == -1 ? 1 : dims[weight[name].perChannelAxis];
		            weight[name].perChannelsConfigs.resize(k);
		            weight[name].zeros.resize(k);
		            weight[name].scales.resize(k);
		            for (int i = 0; i < k; i++) {
			            float minValue = buffer.ReadFloat();
			            float maxValue = buffer.ReadFloat();
			            weight[name].perChannelsConfigs[i] = LowBitConfig(minValue, maxValue, bit, 0);
			            weight[name].zeros[i] = weight[name].perChannelsConfigs[i].zeroPoint;
			            weight[name].scales[i] = weight[name].perChannelsConfigs[i].scale;
		            }
                    buffer.ReadBytes(weight[name].cpuData, weight[name].bytes);
	            }

            printf("Load (%d / %d) \r", (i + 1), weightLen);
            fflush(stdout);
        }
        printf("\n");
        fflush(stdout);
    }

    Data &WeightMap::operator[](const std::string &key) {
        return weight[key];
    }

    void WeightMap::PerChannelQuantizationMultiThread(int st, int end, int m,
                                           float *f, uint8_t *u8, LowBitConfig *configs, int bit) {
        int type = (bit == 4) ? 1 : 0;
        for (int i = st; i < end; i++) {
            float minValue = 1e9, maxValue = -1e9;
            for (int j = 0; j < m; j++) {
                minValue = std::min(minValue, f[i * m + j]);
                maxValue = std::max(maxValue, f[i * m + j]);
            }
            if (bit == 8) {
                configs[i] = LowBitConfig(minValue, maxValue, 8, type);
                for (int j = 0; j < m; j++) {
                    u8[i * m + j] = configs[i].quantization(f[i * m + j]);
                }
            } else {
                configs[i] = LowBitConfig(minValue, maxValue, 4, type);
                for (int j = 0; j < m; j++) {
                    int id = (i * m + j) / 2;
                    uint8_t value = configs[i].quantization(f[i * m + j]);
                    if ((i * m + j) % 2) {
                        u8[id] = (u8[id] & 0xF0) | value;
                    } else {
                        u8[id] = (u8[id] & 0xF) | (value << 4);
                    }
                }
            }
        }
    }

    void WeightMap::SaveLowBitModel(const std::string &fileName, int bit) {
        AssertInXLLM(fileName != "", "Error: output's name shouldn't be empty.\n");
        AssertInXLLM(bit == 0 || bit == 4 || bit == 8 || bit == 16, "Error: only support 16 bit or 8 bit or 4 bit model.\n");
        FileWriter buffer(fileName);

        buffer.WriteInt((int)params.size());
        for (auto &it : params) {
            buffer.WriteString(it.first);
            buffer.WriteString(it.second);
        }

        // 写入权重
        int weightnum = 0;
        for (auto &it : weight) {
            weightnum += (it.second.dims.size() > 0);
        }
        buffer.WriteInt(weightnum);
        int tot = 0;
        for (auto &it : weight) {
            if (it.second.dims.size() == 0) {
                ErrorInXLLM("weight dim is null!");
            }
            buffer.WriteString(it.first);  // name
            Data &data = it.second;
            std::vector<std::string> substrings = {"self_attn.q_proj.weight", "self_attn.k_proj.weight", 
            "self_attn.v_proj.weight", "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
            "mlp.down_proj.weight", "lm_head.weight"};
            if (it.first == "model.embed_tokens.weight")
                data.weightType = WeightType::EMBEDDING;
            else if (containsSubstring(it.first, substrings))
                data.weightType = WeightType::LINEAR;
            buffer.WriteInt((int)data.dims.size());
            for (int i : data.dims) {
                buffer.WriteInt(i);
            }

            if (bit == 0) {  // 不做量化
                DataType dataType = data.dataType;
                if (dataType == DataType::FLOAT32 || dataType == DataType::BFLOAT16 || dataType == DataType::FLOAT16) {
                    buffer.WriteInt((int) dataType);
                    buffer.WriteBytes(data.cpuData, data.bytes);
                } else if (dataType == DataType::INT8 || dataType == DataType::INT4 || dataType == DataType::INT4_NOZERO) {
                    buffer.WriteInt((int) dataType);
                    buffer.WriteInt(data.perChannelAxis);
                    int k = data.perChannelAxis == -1 ? 1 : data.dims[data.perChannelAxis];
                    for (int i = 0; i < k; i++) {
                        buffer.WriteFloat(data.perChannelsConfigs[i].min);
                        buffer.WriteFloat(data.perChannelsConfigs[i].max);
                    }
                    buffer.WriteBytes(data.cpuData, data.bytes);
                } else {
                    ErrorInXLLM("unknown datatype");
                }
            } else { // 做量化
                if (data.weightType == WeightType::NONE) {
                    // 非Embedding/Linear权重，直接写入浮点数据
                    buffer.WriteInt((int) DataType::FLOAT32);
                    buffer.WriteBytes(data.cpuData, data.bytes);
                } else if (data.weightType == WeightType::EMBEDDING) {
                    // Embedding权重，存储成BF16
                    buffer.WriteInt((int) DataType::BFLOAT16);
                    int counts = data.Count(0);
                    std::vector<uint16_t> uDatas;
                    uDatas.resize(counts);
                    for (int i = 0; i < counts; i++) {
                        uDatas[i] = ((uint16_t *) data.cpuData)[i * 2 + 1]; // 只取高16位，因为低16位（float32的有效数字位）是0
                    }
                    buffer.WriteBytes((uint8_t *) uDatas.data(), counts * sizeof(uint16_t));
                } else if (data.weightType == WeightType::LINEAR) {
                    if (bit == 16) {
                        // fp16, 直接转换
                        buffer.WriteInt((int) DataType::FLOAT16);
                        int len = data.Count(0);
                        std::vector<uint16_t> uDatas;
                        uDatas.resize(len);
                        for (int i = 0; i < len; i++) {
                            uDatas[i] = float_to_half(((float *) data.cpuData)[i]);
                        }
                        buffer.WriteBytes((uint8_t *) uDatas.data(), len * sizeof(uint16_t));
                    } else {
                        // Linear层权重，按行量化
                        int k = data.dims[0], m = data.dims[1];
                        int threadNum = GetThreads();
                        int per = k / threadNum;   // 每个线程量化per行
                        int cur = 0;
                        auto pool = GetPool();
                        std::vector <std::future <void> > futures;
                        std::vector<LowBitConfig> configs;
                        std::vector<uint8_t> uDatas;
                        configs.resize(k);

                        int bytes = k * m;
                        if (bit == 4) {
                            bytes = (k * m + 1) / 2;
                        }
                        uDatas.resize(bytes);
                        for (int i = 0; i < threadNum; i++) {
                            int end = cur + per;
                            if (i == threadNum - 1) {
                                end = k;
                            }
                            futures.push_back(pool->enqueue(PerChannelQuantizationMultiThread, cur, end, m,
                                                              (float *) data.cpuData, uDatas.data(), configs.data(),
                                                              bit));
                            cur = end;
                        }
                        for (int i = 0; i < threadNum; i++) {
                            futures[i].get();
                        }

                        buffer.WriteInt(bit == 8 ? (int) DataType::INT8 : (int) DataType::INT4_NOZERO);
                        buffer.WriteInt(0); // 按通道0分通道量化
                        for (int i = 0; i < k; i++) {
                            buffer.WriteFloat(configs[i].min);
                            buffer.WriteFloat(configs[i].max);
                        }
                        buffer.WriteBytes(uDatas.data(), bytes);
                    }
                }
            }
            printf("output (%d / %d)\r", ++tot, weightnum);
            fflush(stdout);
        }
        printf("\n");
        return;
    }

}