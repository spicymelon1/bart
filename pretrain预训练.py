from model_utils.pre_data import PreTrainDataset, loadData, MLM_Data #自定义的读数据的函数
from torch.utils.data import DataLoader, Dataset
from model_utils.models import preModel
import logging        #日志？？代替打印的作用
import os
from model_utils.config import parse_args
from model_utils.utils import setup_device, setup_seed, setup_logging, build_optimizer
import torch
import time
# os.environ['CUDA_VISIBLE_DEVICES']='0'

# 与finetine.py不同处：更注重训练过程中的细节，如每个 step 的损失和剩余时间的记录，但没有验证过程。
def train_and_validate(args):
    # 1. load data  model
    model = preModel(args)     #加载预训练模型
    optimizer, scheduler = build_optimizer(args, model)     #优化器设置，学习率调整
    # model = model.to(args.device)
    use_pre = False

    # 下面不重要（预训练中断的处理，单卡多卡，，，）
    if use_pre: #如果有训练好的模型就直接从保存路径加载来用
        checkpoint = torch.load(args.pre_file, map_location='cpu')
        new_KEY = model.load_state_dict(checkpoint['model_state_dict'],strict=False) #不同1：strict=False，表示在加载模型权重时，允许模型的结构与预训练模型的结构不完全一致
    if args.device == 'cuda': #选择数据串并行训练
        if args.paral == True:
            model = torch.nn.parallel.DataParallel(model.to(args.device))
        else:
            model = model.to(args.device)
        # model = BalancedDataParallel(16, model, dim=0).to(args.device)
    # model = model.to(args.device)
    #-------ema here-----------------

    # 数据部分
    all_data = loadData(args.data_path)
    train_MLM_data = MLM_Data(all_data, args)

    train_dataloader = DataLoader(train_MLM_data, batch_size=args.batch_size, shuffle=True,collate_fn=train_MLM_data.collate) #创建了训练数据集
    # 下面三行不重要
    step = 0
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs

    # 开始训练了！！！找前向传播和梯度回传在哪里
    for epoch in range(args.max_epochs):
        for batch in train_dataloader:
            model.train()
            loss= model(batch) #这里batch里面就是数据！！！调试可以看到是长度为4的list
            loss = loss.mean() #多卡训练取均值
            loss.backward() #loss回传
            optimizer.step() #优化器更新
            optimizer.zero_grad() #优化器清零
            scheduler.step() #学习率调整

            step += 1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}")

        logging.info(f"VAL_Epoch {epoch} step {step}: loss {loss:.3f}")
        # 不同2：预训练不验证，并且模型经过一些轮次就保存一次，不是保存最优模型
        if epoch % 5 == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict()},
                       f'{args.savedmodel_path}/lr{args.learning_rate}epoch{epoch}loss{loss:.3f}pre_model.bin')

def main():
    args = parse_args()           #设置(字典)，单步执行进去，可以看到所有的属性都存放在里面，需要什么复制什么即可
    setup_logging()
    setup_device(args)
    setup_seed(args)
    os.makedirs(args.savedmodel_path, exist_ok=True) #exist_ok=True是在文件夹存在时也不报错，转到声明里面看代码
    logging.info("Training/evaluation parameters: %s", args) #打印所有设置情况，主要是LINUX需要，因为模型训练常用Linux的服务器(没有图形界面)。args用于接收命令行的参数，不用打开vim编辑器一个一个找，直接在命令行就可以设置参数
    train_and_validate(args) #进入模型训练


if __name__ == '__main__':
    main()