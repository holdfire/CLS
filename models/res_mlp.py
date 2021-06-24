# coding=utf-8
"""
timm: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
ResMLP papar: https://arxiv.org/abs/2101.03697
"""


import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from termcolor import cprint
from timm.models import mlp_mixer



class RES_MLP(nn.Module):
    def __init__(self, pretrained=False, dim=128, num_classes_end=2):
        super(RES_MLP, self).__init__()
        self.encoder = mlp_mixer.resmlp_12_224(pretrained=pretrained)
        self.dropout = nn.Dropout(0.1)
        self.GELU = nn.GELU()
        self.linear = nn.Linear(1000, num_classes_end)

    def forward(self, x):
        embedding = self.encoder(x)
        x = self.dropout(embedding)
        x = self.linear(self.GELU(x))
        return embedding, x

def Res_MLP(pretrained):
    net = RES_MLP(pretrained)
    return net
    


if __name__ == "__main__":
    # test code
    pretrained = True
    net = Res_MLP(pretrained)
    print(net)

    input = torch.randn(1, 3, 224, 224)
    output = net(input)
    print(output[0].shape)
    print(output[1].shape)