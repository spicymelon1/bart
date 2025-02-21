#处理字典的文件

import sys
import torch
from collections import Counter #collections计数工具
from transformers import BertTokenizer
from transformers import BartConfig
from transformers import BartForConditionalGeneration
from model_utils.config import parse_args

args = parse_args()         #设置 ，字典， 属性类  config  {}

# 1、读数据
def load_data(path):
    # 打开数据文件，20000行数据
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    datas = []
    # 取出一行，每行两个","
    for line in lines:
        line = line.strip().split(",") #line.strip()去掉换行符，.split(",")按逗号分隔
        if len(line) == 3:
            # 训练集
            text, target = line[1].split(" "), line[2].split(" ") #line[?].split(" ")进一步拆分，逐字提取
            datas.append(text + target) #提取出来后加入datas中
        else:
            text = line[1].split(" ")
            datas.append(text)
    return datas

train_data = load_data('./data/train.csv')

# 2、统计所有出现过的数字
token2count = Counter()     #计数工具 哈希表
for i in train_data:
    token2count.update(i)       #不需要知道原理(调试看见统计了不重复出现的数字个数以及每个数字出现了多少次)

# 把数字从count中取出来变成列表
tail = []
ct = 0 #阈值
for k, v in token2count.items():
    if v >= ct: #超过阈值就加入列表
        tail.append(k)
tail.sort()
vocab = tail

# 3、处理词表：建立自己的词表
vocab.insert(0,"[PAD]")
vocab.insert(100,"[UNK]")
vocab.insert(101,"[CLS]")
vocab.insert(102,"[SEP]")
vocab.insert(103,"[MASK]")
vocab.insert(104,"[EOS]")

# 3、处理词表：在原词表中加字
# 在词表中查询，如果没有就加进去
# tokenizer = BertTokenizer.from_pretrained(args.pre_model_path)
# vocabs = tokenizer.get_vocab()   #获取模型词表
# print(len(vocabs))
# # 建立新词表
# new_vocabs = list(vocabs.keys())
# count = 0
# for v in vocab:         #mn复杂度
#     if v not in vocabs:
#         count += 1
#         new_vocabs.append(v)
#
# print(len(new_vocabs))
new_vocabs = vocab

# 4、保存新的词表
with open(args.pre_model_path+'/vocab.txt', 'w', encoding='utf-8') as f: #词表在mybart_base_chinese下面
    for v in new_vocabs:
        f.write(f"{v}\n")    #保存

# 4、模型部分：为什么词表变了，模型就要变
model = BartForConditionalGeneration.from_pretrained(args.pre_model_path) #原模型Embedding(1297, 768)，表示词汇表大小为 1297，lm_head(768, 1297)
model.resize_token_embeddings(len(new_vocabs)) #新模型Embedding(51440, 768)，表示词汇表大小为 51440，lm_head(768, 51440)
state_dict = model.state_dict()
torch.save(state_dict, args.pre_model_path+'/pytorch_model.bin') #保存新模型
bartconfig = BartConfig.from_pretrained(args.pre_model_path) #保存config，为什么也要：因为有词表的长度设置
bartconfig.vocab_size = len(new_vocabs)
bartconfig.save_pretrained(args.pre_model_path) #新config的vocab_size从1297变成了51440
