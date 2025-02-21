from transformers import AutoTokenizer, BartForConditionalGeneration, AutoModelForMaskedLM

import torch.nn as nn
import torch
class myModel(nn.Module):
    def __init__(self, args):
        super(myModel, self).__init__()
        # self.model = BartModel.from_pretrained(args.pre_model_path)
        self.model = BartForConditionalGeneration.from_pretrained(args.pre_model_path) #从预训练加载的Bart模型
        self.tokenizer = AutoTokenizer.from_pretrained(args.pre_model_path)
        self.pad_id = args.pad_id
        self.tgt_pad_id = args.tgt_pad_id
        self.max_l = args.output_l
        self.beam = args.beam
        self.length_penalty = args.length_penalty
        self.no_repeat = args.no_repeat #no_repeat是什么？？
        self.device = args.device

    #生成bart的输入
    def build_bart_inputs(self, input, tgt=None): #生成mask
        input_mask = (input != self.pad_id)
        if tgt == None:
            return input_mask,None
        else:
            tgt_mask = (tgt != self.tgt_pad_id)
            return input_mask, tgt_mask

    def forward(self, inputs, tgts=None):
        input_mask, tgt_mask = self.build_bart_inputs(inputs, tgts) #
        if tgts == None: #没有target#测试路径，即生成模式(里面的架构与transformer类似)，串行进行
            return self.model.generate(inputs,
                                       max_length=self.max_l,
                                       attention_mask=input_mask,
                                       min_length=2,
                                       num_beams=self.beam,
                                       length_penalty=self.length_penalty,
                                       no_repeat_ngram_size=self.no_repeat,
                                       decoder_start_token_id=102
                                       # early_stopping=True,
                                       )
        outputs = self.model(input_ids=inputs, attention_mask=input_mask,
                             decoder_input_ids=tgts, decoder_attention_mask=tgt_mask)      #训练路径，并行进行
        return outputs.logits #调试可以看到logits的结构是Tensor[2, 80, 1297],表示两个样本，每个样本长度80，对80个数据都做一次1297的分类


class preModel(nn.Module):
    def __init__(self, args):
        super(preModel, self).__init__()
        # self.model = BartModel.from_pretrained(args.pre_model_path)
        self.model = AutoModelForMaskedLM.from_pretrained(args.pre_model_path)  #BART   transformer
        print(self.model)

    # inputs里面的三个东西在MLM_Data类的collate里面产生的！！
    def forward(self, inputs, tgts=None):
        input_ids, attention_mask, labels = inputs["input_ids"], inputs["attention_mask"], inputs["labels"]
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels) #把input_ids、attention_mask和label全部传给模型，这里生成模型不需要token_type_ids(即seq_ids，因为全部都是一句话)
        return outputs.loss
