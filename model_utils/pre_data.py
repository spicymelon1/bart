
import json
import glob
import cv2
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer
import numpy as np
from PIL import Image
import random
import time
import csv
import traceback
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import AutoTokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def loadData(path):
    # 读入数据
    train_data_path = path+"/pro_train_data.csv"
    val_data_path = path + "/pro_val_data.csv"
    test_data_path = path + "/preliminary_a_test.csv"
    path_list = [train_data_path, val_data_path,test_data_path]
    all_data = []
    for index,path in enumerate(path_list):
        with open(path,"r") as f:
            csv_data = csv.reader(f)
            for i in csv_data:
                if len(i)==0:#防止空行
                    break
                if len(i)==3:#训练集 验证集，长度为3的列表：id、ct、医生描述
                    id, input, target=i
                    input=input.split(' ')
                    target=target.split(' ')
                else:#测试集，直接转为id形式
                    id, input,target=i[0], i[1], -1
                    input=input.split(' ')
                if index == 0: #index=0，就是读的训练集
                    all_data.append(input)
                    all_data.append(target)
                else:
                    all_data.append(input)    #验证和测试
    return all_data #调试看到有41000行数据

class PreTrainDataset(Dataset):
    def __init__(self, data_file, input_l, output_l, sos_id=1, eos_id=2, pad_id=0):
        with open(data_file, 'r') as fp:
            reader = csv.reader(fp)
            self.samples = [row for row in reader]
            self.input_l = input_l
            self.output_l = output_l
            self.sos_id = sos_id
            self.pad_id = pad_id
            self.eos_id = eos_id
    def __len__(self):
        return len(self.samples)
    def _try_getitem(self, idx):
        source = [int(x) for x in self.samples[idx][1].split()]
        if len(source)<self.input_l:
            source.extend([self.pad_id] * (self.input_l-len(source)))
        if len(self.samples[idx])<3:
            return np.array(source)[:self.input_l]
        target = [self.sos_id] + [int(x) for x in self.samples[idx][2].split()] + [self.eos_id]
        if len(target)<self.output_l:
            target.extend([self.pad_id] * (self.output_l-len(target)))
        return np.array(source)[:self.input_l], np.array(target)[:self.output_l]



def paddingList(ls:list,val,returnTensor=False):
    ls=ls[:]#不要改变了原list尺寸
    maxLen=max([len(i) for i in ls])
    for i in range(len(ls)):
        ls[i]=ls[i]+[val]*(maxLen-len(ls[i]))
    return torch.tensor(ls,device=device) if returnTensor else ls

def truncate(a:list,b:list,maxLen):
    maxLen-=3#空留给cls sep sep
    assert maxLen>=0
    len2=maxLen//2#若为奇数，更长部分给左边
    len1=maxLen-len2
    #一共就a超长与否，b超长与否，组合的四种情况
    if len(a)+len(b)>maxLen:#需要截断
        if len(a)<=len1 and len(b)>len2:
            b=b[:maxLen-len(a)]
        elif len(a)>len1 and len(b)<=len2:
            a=a[:maxLen-len(b)]
        elif len(a)>len1 and len(b)>len2:
            a=a[:len1]
            b=b[:len2]
    return a,b

class MLM_Data(Dataset):
    #传入句子对列表
    def __init__(self, data, args):
        super().__init__()
        self.data=data
        self.maxLen= args.input_l-3 #规定的最大长度
        self.tk=AutoTokenizer.from_pretrained(args.pre_model_path) #tokenizer
        self.spNum=len(self.tk.all_special_tokens) #特殊字符的数量
        self.tkNum=self.tk.vocab_size #词表长度

    def __len__(self):
        return len(self.data)

    def random_mask(self, text_ids): #看输入是什么？ids； 输出是什么？x和y
        input_ids, output_ids = [], []
        rands = np.random.random(len(text_ids))
        idx=0
        while idx<len(rands):
            if rands[idx]<0.15:#需要mask
                ngram=np.random.choice([1,2,3], p=[0.7,0.2,0.1])#若要mask，进行x_gram mask的概率
                if ngram==3 and len(rands)<7:#太大的gram不要应用于过短文本
                    ngram=2
                if ngram==2 and len(rands)<4:
                    ngram=1
                L=idx+1
                R=idx+ngram#最终需要mask的右边界（开）
                while L<R and L<len(rands):
                    rands[L]=np.random.random()*0.15#强制mask
                    L+=1
                idx=R
                if idx<len(rands):
                    rands[idx]=1#禁止mask片段的下一个token被mask，防止一大片连续mask
            idx+=1

        for r, i in zip(rands, text_ids):
            if r < 0.15 * 0.8:
                input_ids.append(self.tk.mask_token_id)
                output_ids.append(i)#mask预测自己
            elif r < 0.15 * 0.9:
                input_ids.append(i)
                output_ids.append(i)#自己预测自己
            elif r < 0.15:
                input_ids.append(np.random.randint(self.spNum,self.tkNum))
                output_ids.append(i)#随机的一个词预测自己，随机词不会从特殊符号中选取，有小概率抽到自己
            else:
                input_ids.append(i)
                output_ids.append(-100)#保持原样不预测

        return input_ids, output_ids

    # 取数据了！！！
    #耗时操作在此进行，可用上多进程
    def __getitem__(self, item):
        text= self.data[item]#预处理，mask等操作
        text_ids = self.tk.convert_tokens_to_ids(text) #取到一句话先转为ids
        text_ids, out_ids = self.random_mask(text_ids) #random_mask就到了MLM对数据进行的随机遮盖
        input_ids = [self.tk.cls_token_id] + text_ids + [self.tk.sep_token_id] #input_ids要加上一些东西
        token_type_ids=[ 0 ]*(len(text_ids)+2) #[0]就是第一句话，*(len(text_ids)+2)就是只有一句话(这句代码就是segment_embeddings的处理，到bert的ppt里面看)
        labels = [-100] + out_ids + [-100] #label要加上一些东西
        assert len(input_ids)==len(token_type_ids)==len(labels)
        return {'input_ids':input_ids,'token_type_ids':token_type_ids,'labels':labels}

    # 模型的输入(preModel类的inputs参数)就是在这进行转换的
    @classmethod
    def collate(cls,batch):
        input_ids=[i['input_ids'] for i in batch]
        token_type_ids=[i['token_type_ids'] for i in batch]
        labels=[i['labels'] for i in batch]
        input_ids=paddingList(input_ids,0,returnTensor=True)
        token_type_ids=paddingList(token_type_ids,0,returnTensor=True)
        labels=paddingList(labels,-100,returnTensor=True)
        attention_mask=(input_ids!=0)
        return {'input_ids':input_ids,'token_type_ids':token_type_ids
                ,'attention_mask':attention_mask,'labels':labels}

