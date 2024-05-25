#include <gtest/gtest.h>
#include "param.h"

using namespace xllm;

TEST(test_tokenizer, Encode1) {
    Tokenizer tokenizer("/root/autodl-tmp/tokenizer.bin");
    std::vector<float> data = tokenizer.Encode("hello world!");
    ASSERT_EQ(data.size(), 3);
    ASSERT_EQ(data[0], 12199);
    ASSERT_EQ(data[1], 3186);
    // ASSERT_EQ(tokens[2], 36);
}

TEST(test_tokenizer, Encode2) {
    Tokenizer tokenizer("/root/autodl-tmp/tokenizer.bin");
    std::vector<float> data = tokenizer.Encode("hello world!", true);
    ASSERT_EQ(data.size(), 4);
    ASSERT_EQ(data[0], 1);
    ASSERT_EQ(data[1], 12199);
    ASSERT_EQ(data[2], 3186);
    // ASSERT_EQ(data[2], 36);
}

TEST(test_tokenizer, Encode3) {
    Tokenizer tokenizer("/root/autodl-tmp/tokenizer.bin");
    std::vector<float> data = tokenizer.Encode((std::string("[INST] ") + "what is the recipe of mayonnaise?" + " [/INST]"), true);
    ASSERT_EQ(data.size(), 18);
    ASSERT_EQ(data[0], 1);
    ASSERT_EQ(data[2], 25580);
    ASSERT_EQ(data[3], 29962);
    ASSERT_EQ(data[4], 825);
    ASSERT_EQ(data[5], 338);
    ASSERT_EQ(data[6], 278);
    ASSERT_EQ(data[7], 9522);
    ASSERT_EQ(data[8], 412);
    ASSERT_EQ(data[9], 310);
}


TEST(test_utils, trim) {
    ASSERT_EQ(trim(" abc"), "abc");
    ASSERT_EQ(trim(" abc "), "abc");
    ASSERT_EQ(trim("abc "), "abc");
    ASSERT_EQ(trim("a bc "), "a bc");
}
