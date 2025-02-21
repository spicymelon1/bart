# -*- coding: utf-8 -*-
'''这是生成任务的微调过程？？？'''
import logging
import os
import time
import torch
from transformers import PretrainedBartModel
from model_utils.config import parse_args
from model_utils.data import create_dataloaders
from model_utils.models import myModel
from model_utils.score import CiderD, CE
from model_utils.utils import setup_device, setup_seed, setup_logging, build_optimizer,array2str
from torch.cuda.amp import autocast as ac
from tqdm import tqdm as tqdm

os.environ['CUDA_VISIBLE_DEVICES']='0'

# 不用完全理解，关键是哪一块在做什么就行(实际上只有model和loader是自己写的)，知道了以后再用到，复制就行
def validate(model, loader, args, output_file=None, beam=1, n=-1):
    res, gts = [], {}
    tot = 0
    for (source, targets) in tqdm(loader):
        if n>0 and tot>n:
            break
        source = source.cuda() #把x放到cuda上面
        pred = model(source[:, :args. input_l]) #进行预测
        pred = pred.cpu().detach().numpy() #把预测值从 GPU 移动到 CPU，并将其转换为 NumPy 数组
        #print(pred.shape)
        for i in range(pred.shape[0]): # 把预测值数组做成字典
            # res.append({'image_id':tot, 'caption': [array2str(pred[i][2:], args)]})
            # gts[tot] = [array2str(targets[i][1:], args)]
            res.append({'image_id':tot, 'caption': [array2str(pred[i], args)]}) #字典内容是id和医生描述，array2str就是把矩阵元素都变成字符串形式，输入本来是token转化成input_ids之后生成x和y，所以要把输出转换成带空格的字符串    #单步进去看
            gts[tot] = [array2str(targets[i][1:], args)] #标签也要转换成一句话
            tot += 1
    CiderD_scorer = CiderD(df='corpus', sigma=15) #这一步就是把res和gts的描述求相似度
    cider_score, cider_scores = CiderD_scorer.compute_score(gts, res)
    return cider_score

#与pretrain.py不同处：更注重验证过程中的 cider_score，并根据验证结果保存模型，同时在训练过程中明确将数据移动到 GPU
# 训练函数！！！
def train_and_validate(args):
    # 1. load data
    train_dataloader, val_dataloader = create_dataloaders(args) #加载数据
    model = myModel(args)

    #是否使用预训练模型，如果还没有预训练就设为false
    use_pre = False #已经训练过了就设为True
    if use_pre: #加载预训练过的模型
        print('use_pre')
        checkpoint = torch.load(args.my_pre_model_path, map_location='cpu')
        new_KEY = model.load_state_dict(checkpoint['model_state_dict'],strict=True) #不同1：strict=True，表示在加载模型权重时，要求模型的结构与预训练模型的结构完全一致

    optimizer, scheduler = build_optimizer(args, model)
    model = model.to(args.device)
    #-------ema here-----------------

    #进入训练！！！
    model.train()
    #-------------------------------
    # loss, results = validate(model, val_dataloader)
    # 3. training
    step = 0
    best_score = args.best_score     #评估指标，类似分类任务里面的准确率

    # 开始训练了！！！找前向传播和梯度回传在哪里
    for epoch in range(args.max_epochs):
        for (source, targets) in tqdm(train_dataloader): #读数据
            source = source.cuda() #不同2：将输入数据移动到 GPU
            targets = targets.cuda()
            # 训练模式
            model.train()
            pred = model(source[:, :args. input_l], targets[:, :args.output_l]) #得到预测值，source[:, :args. input_l]的第一个":"是样本数，第二个":"是输入长度不能超过input_l
            loss  = CE(pred[:, :-1], targets[:, 1:]) #求loss，targets里面去掉第一个(调试可以看到每个target第一个都是101，这是之前补的，所以要去掉)，pred里面去掉最后一个(因为target和pred的长度要一致，而且最后一个一般都是padding这种，所以去掉最后一个)
            loss = loss.mean() #多卡训练取均值
            loss.backward() #loss回传
            optimizer.step()
            model.zero_grad()
            scheduler.step()
            step += 1
        # 验证
        if epoch % 1 == 0: #恒成立：每一轮都要做验证
            # cider_score？？？
            cider_score = validate(model, val_dataloader, args)
            logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, cider_score {cider_score}")
            if cider_score >= best_score: #不同3：注重验证过程中的 cider_score，并根据验证结果保存模型
                best_score = cider_score
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()},
                        f'{args.savedmodel_path}/model_epoch_{epoch}_cider_score_{cider_score}.bin')



def main(): #和pretrain.py的代码一模一样
    args = parse_args()
    setup_logging()
    setup_device(args) #为什么设置之后就变成cpu了？？？
    setup_seed(args)
    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)
    train_and_validate(args)


if __name__ == '__main__':
    main()
