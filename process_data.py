#处理数据的文件

import pandas as pd  #处理表格数据

pre_train_file= "data/train.csv"

train_df = pd.read_csv(pre_train_file,header=None,names=["id","input","tgt"]) #读入数据


print(train_df.head())

# 划分训练集和验证集
# 使用sample方法，frac采样比例、random_state随机种子、axis轴
train_data = train_df.sample(frac=0.9, random_state=0, axis=0)   #采样0.9的比例
val_data = train_df[~train_df.index.isin(train_data.index)] #train_data.index是取到的数据的下标，train_df.index是全部数据的下标，isin包含，~取反。可以看到结果有2000条数据

# 将数据集保存为csv文件
train_data.to_csv("data/pro_train_data.csv", index=False,header=False)
val_data.to_csv("data/pro_val_data.csv", index=False,header=False)
