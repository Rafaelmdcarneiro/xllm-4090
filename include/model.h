#pragma once

#include <memory>
#include <queue>
#include <thread>
#include <mutex>
#include <random>

#include "utils.h"
#include "data.h"
#include "param.h"
#include "xllm.h"

using RuntimeResult = std::function<void(int index, const char* content)>;
using RuntimeResultBatch = std::function<void(int index, std::vector <std::string> &contents)>;

namespace xllm {

    struct LastTokensUnit {
        int tot = 0;
        std::multiset <int> tokenSet;
        std::queue <int> tokenQueue;

        LastTokensUnit () {}

        LastTokensUnit (int tot) {
            Init(tot);
        }

        void Init(int tot) {
            this->tot = tot;
            tokenSet.clear();
            while (tokenQueue.size() > 0) {
                tokenQueue.pop();
            }
        }

        void Push(int id) {
            if (tokenQueue.size() == tot) {
                tokenSet.erase(tokenSet.find(tokenQueue.front()));
                tokenQueue.pop();
            }
            tokenQueue.push(id);
            tokenSet.insert(id);
        }
    };


    struct LastTokensManager {
        std::vector <LastTokensUnit> units;

        LastTokensManager () {}

        LastTokensManager (int batch, int lastN) {
            units.resize(batch);
            for (int i = 0; i < batch; i++) {
                units[i].Init(lastN);
            }
        }
    };

    class LlamaModel {
    public:
        Tokenizer tokenizer;
        WeightMap weight;

        std::string pre_prompt= "";
        const std::string B_INST{"[INST] "}, E_INST{" [/INST]"}, EOS{""};

        ModelArgs params;

        std::vector<std::vector<float> > sin, cos;
        Data sinData, cosData;

        std::map <std::string, int> deviceMap;

        LlamaModel (const std::string &weightPath, const std::string &tokenPath);
        
        // 推理
        int Forward(
                const Data &inputIds, const Data &attentionMask, const Data &positionIds,
                std::vector <std::pair <Data, Data> > &pastKeyValues,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager(), std::vector <float> *logits = nullptr);

        std::vector <int> ForwardBatch(
                int batch,
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector <std::pair <Data, Data> > &pastKeyValues,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector <std::vector <float>*> *logits = nullptr);
                
        std::string Response(const std::vector<float>& input,
                                     RuntimeResult retCb,
                                     const GenerationConfig &generationConfig = GenerationConfig()); // 根据给出的内容回复

        void ResponseBatch(const std::vector<std::vector<float>> &inputs, std::vector<std::string> &outputs,
                           RuntimeResultBatch retCb, const GenerationConfig &generationConfig);

        void WarmUp(); // 预热

        std::vector<float> MakeInput(std::vector<float> &history, int round, const std::string &input); // 根据历史信息和当前输入生成prompt

        void MakeHistory(std::vector<float> &history, int round, const std::string &input, const std::string &output); // 根据当前回复更新history

        int TOPKSampling(Data &logits, int outerOffset,
                    const GenerationConfig &config, const LastTokensUnit &tokens);

        typedef struct {
            float value;
            int index;
        } IndexedFloat;

        static int compare_indexed_float(const void* a, const void* b);
        
        int sample_top_p(uint8_t* probs, const GenerationConfig& generationConfig);

        void printProfiler() {
            for (auto &it : profiler) {
                printf("%s spend %f\n", it.first.c_str(), it.second);
            }
        };

    private:
        std::map <std::string, float> profiler;
    };
}
