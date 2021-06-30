import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
from collections import OrderedDict
from termcolor import cprint

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.special import softmax

from models import build_model
from tools.oulu_utils import accuracy, performances


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('-a', '--arch', metavar='ARCH', default='deit', help='model architecture')
    parse.add_argument('--input-size', default=224, type=int, help='model input size')
    parse.add_argument('--crop-scale', default=1.5, type=float, help='scale to crop a face from raw image')
    parse.add_argument('--ckpt_path', dest = 'ckpt_path', type = str, default='./ckpt/checkpoint_20210629/checkpoint-100.pth.tar')
    parse.add_argument('-b', '--batch-size', default=1, type=int, metavar='N')
    parse.add_argument('-j', '--workers', default=16, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parse.add_argument('--img-root-dir', default="/home/data4/OULU/", type=str, help='The directory saving dataset')
    parse.add_argument('--val-file-path',  type = str, default="./data/list_oulu/p1_dev_list.txt")
    parse.add_argument('--test-file-path', type = str, default="./data/list_oulu/p1_test_list.txt")
    parse.add_argument('--result-val-path', type = str, default="./data/result_oulu/p1_dev_result.txt")
    parse.add_argument('--result-test-path', type = str, default="./data/result_oulu/p1_test_result.txt")
    parse.add_argument('--out', dest = 'out', type = str, default="./data/result_oulu/p1_result.txt")
    args = parse.parse_args()

    # create model
    args.pretrained = False
    model = build_model.build_model(args).cuda()
    _state_dict = clean_dict(model, torch.load(args.ckpt_path)['state_dict'])
    model.load_state_dict(_state_dict)
    model.eval()

    with torch.no_grad():
        ################  val  ###############
        args.file_path = args.val_file_path
        val_loader, val_sampler = build_dataloader(args)
        score_list = []
        
        for i, datas in enumerate(val_loader):
            images = datas[0].cuda(non_blocking=True)
            spoof_label = datas[1].cuda(non_blocking=True)
            bs = spoof_label.size(0)
            embedding, output = model(images)
            output = F.softmax(output, dim=1)
            output = output.squeeze().cpu().numpy()             
            score_list.append('{} {}\n'.format(output[-1], spoof_label[0]))
            if i % 100 == 0:
                print(i)
        with open(args.result_val_path, 'w') as file:
            file.writelines(score_list) 

        ################  test  ###############
        args.file_path = args.test_file_path
        test_loader, test_sampler = build_dataloader(args)
        score_list = []
        
        for i, datas in enumerate(test_loader):
            images = datas[0].cuda(non_blocking=True)
            spoof_label = datas[1].cuda(non_blocking=True)
            bs = spoof_label.size(0)
            embedding, output = model(images)
            output = F.softmax(output, dim=1)
            output = output.squeeze().cpu().numpy()             
            score_list.append('{} {}\n'.format(output[-1], spoof_label[0]))
            if i % 100 == 0:
                print(i)
        with open(args.result_test_path, 'w') as file:
            file.writelines(score_list)



def clean_dict(model, state_dict, insert_args=None):
    _state_dict = OrderedDict()
    # print(list(state_dict.items())[0][0])
    # print(list(model.state_dict().items())[0][0])
    for k, v in state_dict.items():
        k_ = k.split('.')
        assert k_[0] == 'module'
        if insert_args is not None:
            k_.insert(insert_args[0], insert_args[1])
        
        k = '.'.join(k_[1:])

        if k in model.state_dict().keys() and \
           v.size() == model.state_dict()[k].size():
            _state_dict[k] = v
            cprint(' : successfully load {} - {}'.format(k, v.shape), 'green')
        else:
            try:
                _state_dict[k] = model.state_dict()[k]
                cprint(' : ignore {} - {}'.format(k, v.shape), 'yellow')
            except:
                cprint(' : delete {} - {}'.format(k, v.shape), 'yellow')

    for k, v in model.state_dict().items():
        if k in _state_dict.keys():
            continue
        _state_dict[k] = v
    return _state_dict




def build_dataloader(args):
    """
    Interface to build train or val loader 
    """
    test_transform = alb.Compose([
        alb.Resize(224, 224),
        alb.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
        ])

    dataset = Dataset(
        args.file_path,
        args.img_root_dir,
        transform = test_transform,
        input_size = args.input_size,
        crop_scale = args.crop_scale)

    sampler = None
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True) 
    return loader, sampler


class Dataset(data.Dataset):
    def __init__(self, ann_file, img_root_dir, transform=None, input_size=224, crop_scale=1.5):
        self.ann_file = ann_file
        self.img_root_dir = img_root_dir 
        self.transform = transform
        self.input_size = input_size
        self.crop_scale = crop_scale

        self.image_list = []
        self.label_list = []
        self.bbox_list = []
        with open(self.ann_file, 'r') as f:
            self.lines = f.readlines()

            for line in tqdm(self.lines):
                splits = line.strip().split()
                if len(line.strip()) == 0 or len(splits) != 6:
                    continue
                
                image_full_name = os.path.join(self.img_root_dir, splits[0])
                label = splits[1]
                bbox = splits[2:6]
                bbox = [int(x) for x in bbox]

                self.image_list.append(image_full_name)
                self.label_list.append(int(label))
                self.bbox_list.append(bbox)
        self.n_images = len(self.image_list)
        cprint('Build Train Dataset => in dataloader.py: finally get %d images'%(self.n_images), 'green')

    def __getitem__(self, index):
        img_path = self.image_list[index]
        label = self.label_list[index]
        bbox = self.bbox_list[index]

        cropped_img = get_img_crop(img_path, bbox, self.crop_scale)

        if cropped_img is None or cropped_img.shape[0] * cropped_img.shape[1] == 0:
            face = torch.zeros(3, self.input_size, self.input_size)
            return face, int(0)
        
        face = self.transform(image=cropped_img)['image']
        return face, label

    def __len__(self):
        return self.n_images


def get_img_crop(img_path, bbox, scale):
    """
    crop a raw image from bbox.
    bbox is gotton from a face detection model.
    x_min, y_min, face_x, face_y = bbox
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img is None:
        print('reading empty image: ', img_path)
        return None

    h, w = img.shape[:2]
    x_min, y_min, face_x, face_y = bbox
    x_mid = x_min + face_x / 2.0
    y_mid = y_min + face_y / 2.0

    x_min = max(int(x_mid - face_x * scale / 2.0), 0) 
    x_max = min(int(x_mid + face_x * scale / 2.0), w)
    y_min = max(int(y_mid - face_y * scale / 2.0), 0)
    y_max = min(int(y_mid + face_y * scale / 2.0), h)

    return img[y_min:y_max, x_min:x_max, :]



if __name__ == '__main__':
    main()
