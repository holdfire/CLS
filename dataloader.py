#!/usr/bin/env python3
# coding=utf-8

import os
import numpy as np
import random
import cv2
import argparse
from tqdm import tqdm
# from termcolor import cprint
from tqdm import tqdm

import torch
import torch.utils.data as data
from torchvision import transforms
import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2


# train set transform
train_transform = alb.Compose([
    alb.Resize(448, 448),
    alb.Rotate(limit=30),
    alb.Cutout(1, 25, 25, p=0.1),
    alb.RandomResizedCrop(256, 256, scale=(0.5, 1.0), p=0.5),
    alb.Resize(224, 224),
    alb.HorizontalFlip(),
    alb.ColorJitter(0.25, 0.25, 0.25, 0.125, p=0.2),
    alb.ToGray(p=0.1),
    alb.GaussNoise(p=0.1),
    alb.GaussianBlur(blur_limit=3, p=0.05),
    alb.MotionBlur(blur_limit=(10, 20), p=0.2),
    alb.OneOf([
        alb.RandomBrightnessContrast(),
        alb.FancyPCA(),
        alb.HueSaturationValue(),
    ], p=0.7),
    alb.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2(),
])

# test set transform
test_transform = alb.Compose([
    alb.Resize(224, 224),
    alb.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2(),
])


def build_dataloader(args):
    """
    Interface to build train or val loader 
    """
    if args.mode == "train":
        dataset = Dataset(
            args.train_file_path,
            args.img_root_dir,
            transform = train_transform,
            mode = args.mode,
            balance = True,
            input_size = args.input_size,
            crop_scale = args.crop_scale)
    elif args.mode in ["val", "test"]:
        dataset = Dataset(
            args.val_file_path,
            args.img_root_dir,
            transform = test_transform,
            mode = args.mode,
            balance = False,
            input_size = args.input_size,
            crop_scale = args.crop_scale)


    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True) 
    return loader, sampler


class Dataset(data.Dataset):
    def __init__(self, ann_file, img_root_dir, transform=None, mode="None",balance=False, input_size=224, crop_scale=1.5):
        self.ann_file = ann_file
        self.img_root_dir = img_root_dir
        
        self.transform = transform
        self.mode = mode
        self.balance = balance
        self.input_size = input_size
        self.crop_scale = crop_scale

        self.image_list = []
        self.label_list = []
        # self.bbox_list = []
        with open(self.ann_file, 'r') as f:
            self.lines = f.readlines()

            # balance class numbers while training
            if self.balance and self.mode == 'train':
                self.lines = self.__balance_class__()

            for line in tqdm(self.lines):
                splits = line.strip().split(" ")

                ################### Attention: different for input list #################
                assert len(splits) in [0, 2, 6]
                
                image_full_name = os.path.join(self.img_root_dir, splits[0])
                label = splits[1]
                # bbox = splits[2:6]
                # bbox = [int(x) for x in bbox]

                self.image_list.append(image_full_name)
                self.label_list.append(int(label))
                # self.bbox_list.append(bbox)
        self.n_images = len(self.image_list)
        print('Build Train Dataset => in dataloader.py: finally get %d images'%(self.n_images), 'green')


    def __balance_class__(self, rand_seed=2021):
        """
        这个balance最多只能把数目少的类别翻倍，且只针对train set
        """
        reals, fakes = [], []
        for x in self.lines:
            if int(x.split(" ")[1]) == 0:
                fakes.append(x)
            else:
                reals.append(x)
        print('Build Train Dataset => in dataloader.py: original fakes: %d and real: %d '%(len(fakes), len(reals)), 'green')
        np.random.seed(rand_seed)
        if len(fakes) >= len(reals):
            reals.extend(np.random.permutation(reals)[:len(fakes) - len(reals)])
        else:
            fakes.extend(np.random.permutation(fakes)[:len(reals) - len(fakes)])

        self.img_items = reals + fakes
        np.random.shuffle(self.img_items)
        return self.img_items


    def __getitem__(self, index):
        img_path = self.image_list[index]
        label = self.label_list[index]

        cropped_img = cv2.imread(img_path)
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

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



if __name__ == "__main__":

    # test code
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file_path', default='/home/projects/cls-ly/data/train_list/train_20210817.txt', help='train list demo')
    parser.add_argument('--img-root-dir', default='/home/data2/ly_xiaci/') 
    parser.add_argument('--input_size', default=224, type=int, help='model input size')
    parser.add_argument('--mode', default="train", type=str)
    parser.add_argument('--crop_scale', default=2.5, type=float,help='crop scale to crop a face from raw image')
    parser.add_argument('--batch-size', default=16, type=int, metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--distributed', action='store_true',help='Use multi-processing distributed training to launch')
    args = parser.parse_args()

    loader, sampler= build_dataloader(args)
    print(len(loader))
    for i, (face, label) in enumerate(loader):
        print(face.shape)
        print(type(face))
        print(label)
        print(type(label))
        exit(0)
        
