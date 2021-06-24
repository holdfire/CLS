# coding=utf-8
"""
timm: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
RepVGG paper: https://arxiv.org/abs/2101.03697
"""


import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from termcolor import cprint
from timm.models import byobnet



class REP_VGG(nn.Module):
    def __init__(self, pretrained=False, dim=128, num_classes_end=2):
        super(REP_VGG, self).__init__()
        self.encoder = byobnet.repvgg_a2(pretrained=pretrained)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.ReLU()
        self.linear = nn.Linear(1000, num_classes_end)

    def forward(self, x):
        embedding = self.encoder(x)
        x = self.dropout(embedding)
        x = self.linear(self.act(x))
        return embedding, x

def RepVGG(pretrained):
    net = REP_VGG(pretrained)
    return net
    


if __name__ == "__main__":
    # test code
    pretrained = True
    net = RepVGG(pretrained)
    print(net)

    input = torch.randn(1, 3, 224, 224)
    output = net(input)
    print(output[0].shape)
    print(output[1].shape)