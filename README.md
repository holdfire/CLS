# Face Anti-Spoofing 
This repository is an implementation of Face Anti-Spoofing model.

## Environment
+ OS version: Ubuntu 18.04
+ NVIDIA diver version: 465.27
+ Cuda version: 11.3
+ Cudnn version: 
+ Python version: 3.6.9
+ Python packages installation:   
  `pip3 install -i https://mirrors.aliyun.com/pypi/simple  -r requirements.txt`
  

## Train
#### Dataset
The following datasets are stored in Ningbo/k8s machine.   

|Dataset| Image Path | Total Numbers | Notes |
| :---: | :---: | :---: | :---: |
|ImageNet | /home/data4/ILSVRC2012 | ~1.18 M | for test|
| 防伪训练集0423 | list/train_list/train_all_20210423_bbox.txt| ~907 W | 业务数据 |
| 防伪测试集 | /list_72/cbsr_mas_v6_hifi-mask-test_bbox.txt | ~18 W| 业务数据 |


#### Experiments
+ **Backbone**: Vit, DeiT, Deep Vit, CaiT, Token-to-Token ViT, Cross ViT, PiT, LeViT, CvT, Twins SVT, Masked Patch Prediction, Dino, Accessing Attention(visualize attention weights), Swin-Transformer
+ **Loss**: BCE, Center Loss
+ **Optimizer**: Adam-W, SGD

| Training Date |Dataset| Model | Batch Size | Loss | Optimizer | Learning Rate| Epoch | Weight decay| Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 20210514| train_all_20210423_bbox.txt| deit, pretrained | 256 | bce |Adam-W | lr=5e-4, cosine | 10 | 1e-4| BGR, crop=2.5|


## Reference:
#### Code:
+ pytroch-imageNet: https://github.com/pytorch/examples/blob/master/imagenet/main.py
+ pytorch-image-models: https://github.com/rwightman/pytorch-image-models
+ Vit-Pytorch: https://github.com/lucidrains/vit-pytorch  
+ DeiT: https://github.com/facebookresearch/deit
+ Swin-Transformer: https://github.com/microsoft/Swin-Transformer  

  
#### Paper:
+ Vit: https://arxiv.org/abs/2010.11929
+ DeiT: https://arxiv.org/abs/2012.12877
+ Swin: http://arxiv.org/abs/2103.14030
+ CaiT: https://arxiv.org/abs/2103.17239

## Change log:
**20210524**: change the order of input image channel from BGR to RGB, to better adapt to pretrained model in ImageNet.
+ dataloader.py：修改了图片加载顺序BGR->RGB
+ test.py：修改了图片加载顺序BGR->RGB
+ train.py: 在一个epoch内保存模型时，只保存权重
