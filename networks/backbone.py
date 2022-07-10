import torch.nn as nn
import torch
import numpy as np
from transformers import BertModel, BertConfig

class Bert_Encoder(nn.Module):

    def __init__(self, args, out_token=False):
        super(Bert_Encoder, self).__init__()

        # load model
        self.encoder = BertModel.from_pretrained(args.bert_path).cuda()
        self.bert_config = BertConfig.from_pretrained(args.bert_path)

        # the dimension for the final outputs
        self.output_size = args.encoder_output_size
        self.out_dim = self.output_size

        self.linear_transform = nn.Linear(self.bert_config.hidden_size, self.output_size, bias=True)

        self.layer_normalization = nn.LayerNorm([self.output_size])


    def get_output_size(self):
        return self.output_size

    def forward(self, inputs):
        output = self.encoder(inputs)[1]
        
        output = self.linear_transform(output)
        
        return output