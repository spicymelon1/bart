
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import csv
import traceback
from transformers import AutoTokenizer

# class BaseDataset(Dataset):
#     def _try_getitem(self, idx):
#         raise NotImplementedError
#     def __getitem__(self, idx):
#         wait = 0.1
#         while True:
#             try:
#                 ret = self._try_getitem(idx)
#                 return ret
#             except KeyboardInterrupt:
#                 break
#             except (Exception, BaseException) as e:
#                 exstr = traceback.format_exc()
#                 print(exstr)
#                 print('read error, waiting:', wait)
#                 time.sleep(wait)
#                 wait = min(wait*2, 1000)

# 不同！！！！！！
# 以前是在init里面得到x和y，这次只是把？材料？放到了init里面，x和y是在getitem里面的得到
class TranslationDataset(Dataset): #读数据
    def __init__(self, data_file, args):
        with open(data_file, 'r') as fp:
            reader = csv.reader(fp)
            self.samples = [row for row in reader][:16] #只取16个样本数据，每一条数据里面有id、ct和医生描述

            #是否全部数据
            self.input_l = args.input_l       #输入长度
            self.output_l = args.output_l       #输出长度
            self.sos_id = args.sos_id            #开始token
            self.pad_id = args.pad_id            #pad_token，多截少补
            self.eos_id = args.eos_id            # 结束
            self.tgt_pad_id = args.tgt_pad_id       # 结束pad
            self.tk=AutoTokenizer.from_pretrained(args.pre_model_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        source =[self.sos_id]+ self.tk.convert_tokens_to_ids([x for x in self.samples[idx][1].split()]) + [self.eos_id] #根据下标取样本，self.samples[idx][1].split()就是把样本中的x逐字取出变成一个列表，tk.convert_tokens_to_ids把每个数字转换成词表索引，前后加上句子始终的标志
        if len(source)<self.input_l: #x长度不够就补0
            source.extend([self.pad_id] * (self.input_l-len(source)))
        if len(self.samples[idx])<3: #样本长度小于3(即测试集，没有医生描述这一列)，只读x，否则还要读y
            return np.array(source)[:self.input_l]

        target = [self.sos_id] + self.tk.convert_tokens_to_ids([x for x in self.samples[idx][2].split()]) + [self.eos_id] #根据下标取样本，self.samples[idx][1].split()就是把样本中的y逐字取出变成一个列表，tk.convert_tokens_to_ids把每个数字转换成词表索引，前后加上句子始终的标志
        if len(target)<self.output_l: #y长度不够就补0
            target.extend([self.tgt_pad_id] * (self.output_l-len(target)))
        return np.array(source)[:self.input_l], np.array(target)[:self.output_l] #调试可以看到source是长度150的列表，target是长度80的列表
        #

def create_dataloaders(args, test=False):
    if not test: #如果不是测试集，就读取训练集和验证集
        train_data_path = args.data_path+"/pro_train_data.csv"
        val_data_path = args.data_path + "/pro_val_data.csv"
        train_data = TranslationDataset(train_data_path, args)
        valid_data = TranslationDataset(val_data_path, args)

        #num_workers和drop_last是什么？？？
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
        valid_loader = DataLoader(valid_data, batch_size=args.val_batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)

        return train_loader, valid_loader
    else:
        test_data_path = args.data_path + "/preliminary_a_test.csv"
        test_data = TranslationDataset(test_data_path, args)
        test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
        return test_loader