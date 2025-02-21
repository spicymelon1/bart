import torch
import torch.nn as nn
from pathlib import Path
import time
import json
import numpy as np
import random
import cv2
import copy
from collections import OrderedDict
from torch.nn.functional import normalize


def get_parameters(model, pars):
    ret = [{'params': getattr(model, x).parameters()} for x in pars]
    print(ret)
    return ret


def output_tensor(x, precision=3):
    print(np.round(x.detach().cpu().numpy(), precision))


def to_device(data, device):
    if isinstance(data, torch.Tensor):
        data = data.to(device)
    elif isinstance(data, np.ndarray):
        data = to_device(torch.from_numpy(data), device)
    elif isinstance(data, tuple):
        data = tuple(to_device(item, device) for item in data)
    elif isinstance(data, list):
        data = list(to_device(item, device) for item in data)
    elif isinstance(data, dict):
        data = dict((k, to_device(v, device)) for k, v in data.items())
    else:
        raise TypeError('Unsupported Datatype! Must be a Tensor/List/Tuple/Dict.', type(data), data)
    return data


class Smoother():
    def __init__(self, window):
        self.window = window
        self.num = {}
        self.sum = {}

    def update(self, **kwargs):
        """
        为了调用方便一致，支持kwargs中有值为None的，会被忽略
        kwargs中一些值甚至可以为dict，也就是再套一层。
        示例: update(a=1, b=2, c={'c':1, 'd':3})，相当于update(a=1, b=2, c=1, d=3)
        如果值为参数的None的话忽略
        """
        values = {}
        for key in kwargs:
            if isinstance(kwargs[key], dict):
                for x in kwargs[key]:
                    values[x] = kwargs[key][x]  # 有可能会覆盖，如update(a=1,b={'a':2})
            else:
                values[key] = kwargs[key]
        for key in values:
            if values[key] is None:
                continue
            if key not in self.num:
                self.num[key] = []
                self.sum[key] = 0
            self.num[key].append(values[key])
            self.sum[key] += values[key]

            if len(self.num[key]) > self.window:
                self.sum[key] -= self.num[key][-self.window - 1]
            if len(self.num[key]) > self.window * 2:
                self.clear(key)
        pass

    def clear(self, key):
        del self.num[key][:-self.window]

    def value(self, key=None, mean=True):
        if mean:
            if key is None:
                return {key: self.sum[key] / min(len(self.num[key]), self.window) for key in self.num}
            return self.sum[key] / min(len(self.num[key]), self.window)
        if key is None:
            return {key: np.array(self.num[key]) for key in self.num}
        return np.array(self.sum[key])

    def keys(self):
        return self.num.keys()





import logging
import random

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch
from transformers import AdamW, get_linear_schedule_with_warmup


def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count() #统计GPU个数

def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def setup_logging():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger


def build_optimizer(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    large_lr = ['']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #      'weight_decay': args.weight_decay},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    # optimizer_grouped_parameters = [
    #     {'params': [j for i, j in model.named_parameters() if (not 'bert' in i and not any(nd in i for nd in no_decay))],
    #      'lr': args.learning_rate, 'weight_decay': args.weight_decay},
    #     {'params': [j for i, j in model.named_parameters() if (not 'bert' in i and any(nd in i for nd in no_decay))],
    #      'lr': args.learning_rate, 'weight_decay': 0.0},
    #     {'params': [j for i, j in model.named_parameters() if ('bert' in i and not any(nd in i for nd in no_decay))],
    #      'lr': args.learning_rate, 'weight_decay': args.weight_decay},
    #     {'params': [j for i, j in model.named_parameters() if ('bert' in i and any(nd in i for nd in no_decay))],
    #      'lr': args.learning_rate, 'weight_decay': 0.0},
    # ]
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    return optimizer, scheduler

from transformers import AutoTokenizer
def array2str(arr, args):
    tk = AutoTokenizer.from_pretrained(args.pre_model_path) #用分词器转回去
    out = ''
    for i in range(len(arr)): #arr是一个一个元素
        if arr[i]==args.pad_id or arr[i]==args.eos_id:
            break
        if arr[i]==args.sos_id:
            continue
        out += tk.convert_ids_to_tokens([arr[i]])[0] + ' ' #！！！！out是整个一句话
    if len(out.strip())==0:
        out = '0'
    return out.strip()

