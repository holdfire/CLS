#!/usr/bin/env python3
# coding=utf-8

import os
import numpy as np
import random
import cv2
import argparse
from tqdm import tqdm
from termcolor import cprint
from tqdm import tqdm

import torch
import torch.utils.data as data
from torchvision import transforms


def build_train_loader(args):
    train_trans = transforms.Compose([
        # ToTensor(): conver type numpy.ndarray into torch.tensor, and normalize each value to [0, 1]
        transforms.ToTensor(),                     
        transforms.Resize((args.input_size, args.input_size)),
        transforms.RandomHorizontalFlip(),
        # channel order: R, G, B
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        # channel order: B, G, R
    ])

    train_dataset = TrainDataset(
        args.train_list,
        transform = train_trans,
        input_size = args.input_size,
        crop_scale = args.crop_scale)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=(train_sampler is None),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True)
    
    return train_loader, train_sampler


class TrainDataset(data.Dataset):
    def __init__(self, ann_file, transform=None, input_size=224, crop_scale=2.5):
        self.ann_file = ann_file
        self.transform = transform
        self.input_size = input_size
        self.crop_scale = crop_scale

        self.image_list = []
        self.label_list = []
        self.bbox_list = []
        cprint('Build Train Dataset => in dataloader.py: start preparing train dataset from file: %s'%(ann_file), 'green')
        self.init()

    def init(self):
        with open(self.ann_file, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                # line format: image_path  label  x_min  y_min  face_x  face_y
                splits = line.strip().split()
                # using bbox info to crop face
                if len(line.strip()) == 0 or len(splits) != 6:
                    continue
                
                image_full_name = splits[0]
                label = splits[1]
                bbox = splits[2:6]
                bbox = [int(x) for x in bbox]

                self.image_list.append(image_full_name)
                self.label_list.append(int(label))
                self.bbox_list.append(bbox)
        self.n_images = len(self.image_list)
        cprint('Build Train Dataset => in dataloader.py: finish preparing: there are %d image items'%(self.n_images), 'green')
        # Channle order: B, G, R
        # cprint('WARN => in dataloader.py: OpenCV is kept as load engine. The input channels order is kept as B, G, R.', 'yellow')
        # Channle order: R G, B
        cprint('WARN => in dataloader.py: The input channels order is changed to R, G, B.', 'yellow')

    def __getitem__(self, index):
        img_path = self.image_list[index]
        label = self.label_list[index]
        bbox = self.bbox_list[index]

        cropped_img = get_img_crop(img_path, bbox, self.crop_scale)

        if cropped_img is None or cropped_img.shape[0] * cropped_img.shape[1] == 0:
            face = torch.zeros(3, self.input_size, self.input_size)
            return face, int(0)
        
        cropped_img = cropped_img[: ,:, [2, 1, 0]]
        face = self.transform(cropped_img)
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

    if img is None:
        print('reading empty image: ', img_path)
        return None

    shape = img.shape
    h, w = shape[:2]
    x_min, y_min, face_x, face_y = bbox
    x_mid = x_min + face_x / 2.0
    y_mid = y_min + face_y / 2.0

    if x_mid<=0 or y_mid<=0 or face_x<=0 or face_y<=0:
        return None

    x_min = int( max(x_mid - face_x * scale / 2.0, 0) )
    x_max = int( min(x_mid + face_x * scale / 2.0, w) )
    y_min = int( max(y_mid - face_y * scale / 2.0, 0) )
    y_max = int( min(y_mid + face_y * scale / 2.0, h) )

    if x_min >= x_max or y_min >= y_max:
        return None
    
    return img[y_min:y_max, x_min:x_max, :]




if __name__ == "__main__":

    # test code
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_list', default='/home/projects/list/train_list/train_all_20210423_bbox.txt',help='train list demo')
    parser.add_argument('--input_size', default=224, type=int, help='model input size')
    parser.add_argument('--crop_scale', default=2.5, type=float,help='crop scale to crop a face from raw image')
    parser.add_argument('--batch-size', default=16, type=int, metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--distributed', action='store_true',help='Use multi-processing distributed training to launch')
    args = parser.parse_args()

    loader = build_train_loader(args)
    for i, (face, label) in enumerate(loader):
        print(face.shape)
        print(type(face))
        print(label)
        print(type(label))
        exit(0)
        
