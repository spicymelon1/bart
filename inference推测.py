'''这是生成任务的推测过程？？？'''
from tqdm import tqdm
import csv
from model_utils.utils import to_device, array2str
from model_utils.models import myModel
from model_utils.data import create_dataloaders
import torch
from model_utils.config import parse_args


def inference(args):
    test_loader = create_dataloaders(args,test=True) #创建测试集
    model = myModel(args) #加载模型
    print(args.ckpt_file)

    checkpoint = torch.load(args.ckpt_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    model.to('cuda:0')
    model.eval()
    #用测试数据做训练
    fp = open(args.test_output_csv, 'w', newline='')
    writer = csv.writer(fp)
    tot = 0
    for source in tqdm(test_loader):
        source = to_device(source, 'cuda:0')
        pred = model(source)
        pred = pred.cpu().numpy()
        for i in range(pred.shape[0]):
            writer.writerow([tot, array2str(pred[i][2:], args)]) #array2str把输出转换成带空格的字符串
            tot += 1
    fp.close()

if __name__ == '__main__':
    args = parse_args()
    inference(args)