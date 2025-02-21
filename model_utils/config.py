
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Baseline for Weixin Challenge 2022")
    # ========================= Base Configs ==========================
    parser.add_argument("--seed", type=int, default=2025, help="random seed.") #随机种子
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout ratio') #丢弃率
    parser.add_argument('--ema', type=bool, default=False, help='ema') #提升模型准确度的一些trick
    parser.add_argument('--attack', type=str, default='None', help='attack:fgm,pgd')   #提升模型准确度的一些trick
    parser.add_argument('--use_fp16', type=bool, default=False, help='fp16')
    parser.add_argument('--paral', default=False, type=bool, help='is parallel?') #多张显卡
    parser.add_argument('--learning_rate', default=3e-5, type=float, help='initial learning rate')

    # ========================= Generate Configs ==========================
    parser.add_argument('--beam', default=5, type=int, help='beamnum?')
    parser.add_argument('--length_penalty', default=1, type=float, help='length_penalty')
    parser.add_argument('--no_repeat', default=4, type=int, help='no_repeat_ngram_size')

    # ========================= Data Configs ==========================
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--test_output_csv', type=str, default=R'E:\准研究生\李哥项目班\代码\第十节代码\bart分享\data\result.csv')
    parser.add_argument('--val_ratio', default=0.1, type=float, help='split 10 percentages of training data as validation')
    parser.add_argument('--batch_size', default=2, type=int, help="use for training duration per worker")
    parser.add_argument('--val_batch_size', default=2, type=int, help="use for validation duration per worker")
    parser.add_argument('--test_batch_size', default=2, type=int, help="use for testing duration per worker")
    parser.add_argument('--prefetch', default=16, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=1, type=int, help="num_workers for dataloaders")

    # ======================== SavedModel Configs =========================
    parser.add_argument('--savedmodel_path', type=str, default='save/pretrain')
    parser.add_argument('--ckpt_file', type=str, default=R'E:\准研究生\李哥项目班\代码\第十节代码\bart分享\save\pretrain\model_epoch_0_cider_score_6.430476657934441e-05.bin')
    parser.add_argument('--best_score', default=0, type=float, help='save checkpoint if cider > best_score')
    parser.add_argument('--pre_model_path', type=str, default='mybart-base-chinese')
    parser.add_argument('--my_pre_model_path', type=str, default=r'E:\准研究生\李哥项目班\代码\第十节代码\bart练习\save\pretrain\model_epoch_0_cider_score_0.00011678801024382014.bin')


    # ========================= Learning Configs ==========================
    parser.add_argument('--max_epochs', type=int, default=50, help='How many epochs')
    parser.add_argument('--max_steps', default=50000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--print_steps', type=int, default=20, help="Number of steps to log training metrics.")
    parser.add_argument('--warmup_steps', default=1000, type=int, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--minimum_lr', default=0., type=float, help='minimum learning rate')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--device", default="cuda", type=str, help="device")



    # ==========================  BART =============================
    parser.add_argument('--n_layer', type=int, default=6, help="Number of layers.")
    parser.add_argument('--input_l', type=int, default=150, help="Number of layers.")
    parser.add_argument('--output_l', type=int, default=80, help="Number of layers.")
    parser.add_argument('--n_token', type=int, default=2000, help="Number of layers.")
    parser.add_argument('--sos_id', type=int, default=101, help="Number of layers.")
    parser.add_argument('--eos_id', type=int, default=105, help="Number of layers.")
    parser.add_argument('--pad_id', type=int, default=0, help="Number of layers.")
    parser.add_argument('--tgt_pad_id', type=int, default=0, help="Number of layers.")
    parser.add_argument('--bart_dir', type=str, default='/media/lsc/model/bart')
    # parser.add_argument('--bart_learning_rate', type=float, default=5e-5)
    parser.add_argument('--bart_warmup_steps', type=int, default=5000)
    parser.add_argument('--bart_max_steps', type=int, default=30000)
    # parser.add_argument("--bert_hidden_dropout_prob", type=float, default=0.1)
    # parser.add_argument("--bert_output_dim", type=float, default=768)
    # parser.add_argument("--bert_hidden_size", type=float, default=768)
    # parser.add_argument("--bert_seq_length", type=float, default=256)

    # ========================== preTrain =============================



    return parser.parse_args()








