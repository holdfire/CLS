import os
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
from loguru import logger 

import torch
from torch.utils.data import DataLoader
from scipy.special import softmax

from mymodel import ResNet10
from myutils import *
from dataset import MyDataset, test_transform

from test_config import TestConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--result', type=str, default='val_test_result.txt')
    parser.add_argument('--scale', action='store_true', default=False)
    parser_args = parser.parse_args()

    args = TestConfig()
    args.ckpt_path = parser_args.ckpt_path
    args.result = parser_args.result
    args.scale = parser_args.scale

    if args.scale:
        print('scale')

    if not os.path.exists(args.ckpt_path):
        logger.error('ckpt {} not exists.'.format(args.ckpt_path))
    sd = torch.load(args.ckpt_path)['state_dict']

    model = ResNet10().cuda()
    model.load_state_dict(sd)
    model.eval()

    test_dataset = MyDataset(args.img_data_root, args.img_list_root, args.face_data_path, split='val_test', transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10, pin_memory=True, drop_last=False)

    all_logits, all_paths = [], []
    for i, datas in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            images = datas[0].cuda(non_blocking=True)
            paths = datas[2]
            outputs = model(images)

            all_logits.extend(outputs)
            all_paths.extend(paths)

    all_logits = torch.stack(all_logits)

    if not args.scale:
        # all_logits[:, 0] /= 5.0
        # all_logits[:, 1] /= 3.6
        real_logits = all_logits[:, 1] / 3.6
        fake_logits = (all_logits[:, 0] + all_logits[:, 2]) / 5.0
        all_logits = torch.stack([real_logits, fake_logits]).t()

        all_real_probs = torch.softmax(all_logits, dim=1)[:, 0].data.cpu().numpy()
    else:
        all_real_probs = torch.softmax(all_logits, dim=1)[:, 1].data.cpu().numpy()

    with open(args.result, 'w') as f:
        for img_path, real_prob in zip(all_paths, all_real_probs):
            assert img_path.startswith('val') or img_path.startswith('test'), "your image path must start with val or test!"
            
            if args.scale and img_path.startswith('val'):
                real_prob -= 0.07

            img_path = img_path.replace(args.img_data_root, '')
            f.write('{} {}\n'.format(img_path, real_prob))

if __name__ == '__main__':
    main()
