#include "fstream"

#include "model.h"
#include "utils.h"

struct BenchmarkConfig {
    std::string weightPath;
    std::string tokenPath;
    int threads = 4; // 使用的线程数
    int limit = -1; // 输出token数限制，如果 < 0 则代表无限制
    std::string output; // 输出文件，如果不设定则输出到屏幕
};

void Usage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "[-h|--help]:                  显示帮助" << std::endl;
    std::cout << "<--weight> <args>:            模型文件的路径" << std::endl;
    std::cout << "<--token> <args>:             分词器文件的路径" << std::endl;
    std::cout << "<-t|--threads> <args>:        使用的线程数量" << std::endl;
    std::cout << "<-l|--low>:                   使用低内存模式" << std::endl;
    std::cout << "<--top_p> <args>:             采样参数top_p" << std::endl;
    std::cout << "<--top_k> <args>:             采样参数top_k" << std::endl;
    std::cout << "<--temperature> <args>:       采样参数温度，越高结果越不固定" << std::endl;
    std::cout << "<--repeat_penalty> <args>:    采样参数重复惩罚" << std::endl;
}

void ParseArgs(int argc, char **argv, BenchmarkConfig &config) {
    std::vector <std::string> sargv;
    for (int i = 0; i < argc; i++) {
        sargv.push_back(std::string(argv[i]));
    }
    for (int i = 1; i < argc; i++) {
        if (sargv[i] == "-h" || sargv[i] == "--help") {
            Usage();
            exit(0);
        } else if (sargv[i] == "--weight") {
            config.weightPath = sargv[++i];
        } else if (sargv[i] == "--token") {
            config.tokenPath = sargv[++i];
        } else if (sargv[i] == "-t" || sargv[i] == "--threads") {
            config.threads = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "-l" || sargv[i] == "--limit") {
            config.limit = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "-o" || sargv[i] == "--output") {
            config.output = sargv[++i];
        } else {
            Usage();
            exit(-1);
        }
    }
}

int main(int argc, char **argv) {
    BenchmarkConfig config;
    ParseArgs(argc, argv, config);
    xllm::SetThreads(config.threads);
    std::unique_ptr<xllm::LlamaModel> model = std::make_unique<xllm::LlamaModel>(config.weightPath, config.tokenPath);

    xllm::GenerationConfig generationConfig;
    generationConfig.output_token_limit = config.limit;
    xllm::PrintInstructionInfo();

    int promptTokenNum = 0;
    int tokens = 0;
    int round = 0;
    std::string input = "hello";
    std::vector<float> history;
    auto st = std::chrono::system_clock::now();
    std::string ret = model->Response(model->MakeInput(history, round, input), [&tokens](int index, const char* content) {
        if (index >= 0) {
            tokens++;
            printf("%s", content);
            fflush(stdout);
        }
        if (index == -1) {
            printf("\n");
        }
    }, generationConfig);
    float spend = xllm::GetSpan(st, std::chrono::system_clock::now());

    printf("output %d tokens\nuse %f s\nspeed = %f tokens / s\n", tokens, spend, tokens / spend);
    // xllm::PrintProfiler();
    return 0;
}