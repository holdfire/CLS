# coding=utf-8
"""
Reference: https://github.com/lucidrains/vit-pytorch
"""

import torch
import torch.nn as nn
from vit_pytorch import ViT

class VIT(nn.Module):
    def __init__(self, pretrained=False, dim=128, num_classes_end=2):
        super(VIT, self).__init__()
        self.encoder = ViT(
            image_size = 224,
            patch_size = 32,
            num_classes = 128,
            dim = 512,
            depth = 6,
            heads = 16,
            mlp_dim = 1024,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU(inplace=True)
        self.linear = nn.Linear(128, num_classes_end)

        # load pretrained model if needed
        if pretrained:
            cprint("WARN => Attention: using pre-trained model: vit", color="yellow")
            state_dict = load_state_dict_from_url(model_urls["deit_base_distilled_patch16_224"], progress=progress)['state_dict']
            for key in list(state_dict.keys()):
                value = state_dict[key]
                new_key = key.split("encoder_q.")[1]
                state_dict[new_key] = value
                state_dict.pop(key)

            print("in vit.py, line 50")                                    
            print(state_dict.keys())

            _state_dict = self.encoder.state_dict()
            _state_dict.update(state_dict)
            self.encoder.load_state_dict(_state_dict)

    def forward(self, x):
        embedding = self.encoder(x)
        x = self.dropout(embedding)
        x = self.linear(self.relu(x))
        return embedding, x


def vit(pretrained):
    net = VIT(pretrained)
    return net