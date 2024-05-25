import sys
from transformers import AutoModelForCausalLM

import struct
import numpy as np
import torch

def writeString(fo, s):
    fo.write(struct.pack('i', len(s)))
    fo.write(s.encode())

def writeKeyValue(fo, key, value):
    writeString(fo, key)
    writeString(fo, value)

data_type_dict = {
    "int4": 8,
    "int8": 3,
    "float16": 7,
    "float32": 0,
    "bfloat16": 1,
}

v = np.random.randint(-127, 127, [10, 20])
temp = v
c_max = np.expand_dims(np.abs(v).max(axis = -1), -1)
c_scale = c_max / 127.0
v = (v / c_scale + 128.5).clip(1, 255).astype(np.uint8)

def write_int8(fo, v):
    c_max = np.expand_dims(np.abs(v).max(axis = -1), -1).clip(0.1, 1e100)
    c_scale = c_max / 127.0
    v = (v / c_scale + 128.5).clip(1, 255).astype(np.uint8)
    fo.write(struct.pack('i', 3))
    fo.write(struct.pack('i', 0))
    for i in range(c_max.shape[0]):
        fo.write(struct.pack('f', -c_max[i][0]))
        fo.write(struct.pack('f', c_max[i][0]))
    fo.write(v.data)

def write_int4(fo, v):
    c_min = np.expand_dims(-np.abs(v).max(axis = -1), -1)
    c_max = np.expand_dims(np.abs(v).max(axis = -1), -1)
    c_scale = c_max / 7.0
    c_min = c_scale * -8.0
    v = (v - c_min) / c_scale
    v = (v + 0.5).astype(np.int8).clip(0, 15).astype(np.uint8)
    v = v[:, 0::2] * 16 + v[:, 1::2]
    fo.write(struct.pack('i', 8))
    fo.write(struct.pack('i', 0))
    for i in range(c_min.shape[0]):
        fo.write(struct.pack('f', c_min[i][0]));
        fo.write(struct.pack('f', c_max[i][0]));
    fo.write(v.data)

def tofile(exportPath,
           model,
           pre_prompt = None,
           user_role = None,
           bot_role = None,
           history_sep = None,
           dtype = "float16"):
    if (dtype not in data_type_dict):
        print("dtype should in ", list(data_type_dict.keys()))
        exit(0)

    dict = model.state_dict()
    fo = open(exportPath, "wb")

    # 1. model info
    modelInfo = model.config.__dict__
    if model.generation_config is not None:
        modelInfo.update(model.generation_config.__dict__)
    if ("model_type" not in modelInfo):
        print("unknown model_type.")
        exit(0)

    if (pre_prompt):
        modelInfo["pre_prompt"] = pre_prompt
    if (user_role):
        modelInfo["user_role"] = user_role
    if (bot_role):
        modelInfo["bot_role"] = bot_role
    if (history_sep):
        modelInfo["history_sep"] = history_sep

    # # of modelInfo, key-value pairs
    fo.write(struct.pack('i', len(modelInfo)))
    for it in modelInfo.keys():
        writeKeyValue(fo, str(it), str(modelInfo[it]))

    # 2. vocab
    # # of vocab, vocab length, vocab char, ID
    # vocab = tokenizer.get_vocab()
    # fo.write(struct.pack('i', len(vocab)))
    # for v in vocab.keys():
    #     s = v.encode()
    #     fo.write(struct.pack('i', len(s)))
    #     for c in s:
    #         fo.write(struct.pack('i', c))
    #     fo.write(struct.pack('i', vocab[v]))

    # 3. weight
    # # of weights, length of weight name, weight name, tensor.shape, tensor
    weight_type_dict = {}  # weight_name:weight_type(linear,embedding)
    for key, m in model.named_modules():
        if (isinstance(m, torch.nn.Linear)):
            weight_type_dict[key + ".weight"] = "linear"
        if (isinstance(m, torch.nn.Embedding)):
            weight_type_dict[key] = "embedding"
    fo.write(struct.pack('i', len(dict)))
    tot = 0
    for key in dict:
        ori_np_data_type = np.float32
        to_data_type = 0
        # 只对linear层做了量化!
        if (weight_type_dict.get(key, None) == "linear"):
            to_data_type = data_type_dict[dtype]
            if (dtype == "float16"):
                ori_np_data_type = np.float16
        tensor = dict[key].numpy().astype(ori_np_data_type)  # TODO: fp32->fp16有损失，转换成bf16

        fo.write(struct.pack('i', len(key)))
        fo.write(key.encode())
        fo.write(struct.pack('i', len(tensor.shape)))
        for i in tensor.shape:
            fo.write(struct.pack('i', i))
        if (to_data_type == 3):
            write_int8(fo, tensor)
        elif (to_data_type == 8):
            write_int4(fo, tensor)
        else:
            fo.write(struct.pack('i', to_data_type))
            fo.write(tensor.data)   # tensor.data: 指向数组数据的内存缓冲区的指针
        tot += 1
        print("output (", tot, "/", len(dict), end = " )\r")
    print("\nfinish.")
    fo.close()

if __name__ == "__main__":
    cachePath = sys.argv[1]
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True, cache_dir=cachePath).float() # /root/autodl-tmp/hf
    model = model.eval()

    exportPath = sys.argv[2]
    tofile(exportPath, model)
