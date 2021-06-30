#!/usr/bin/env python

import os
import argparse
import yaml


def parse_args():
    parser = argparse.ArgumentParser()

    # basic args
    parser.add_argument('-a', '--arch', metavar='ARCH', default='vit',
                        help='model architecture')
    parser.add_argument('--img-root-dir', default="/home/data4/OULU/", type=str,
                        help='The directory saving dataset')
    parser.add_argument('--train-file-path', default=None, type=str,
                        help='The dataset list path for training')
    parser.add_argument('--val-file-path', default=None, type=str,
                        help='The dataset list path for validation')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--eval-interval-time', default=1, type=int,
                        help='The eval interval time')
    
    # hyper parameters
    parser.add_argument('--input-size', default=224, type=int,
                        help='model input size')
    parser.add_argument('--crop-scale', default=2.5, type=float,
                        help='scale to crop a face from raw image')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-schedule', default='cosine', type=str,
                        help='learning rate schedule')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.05, type=float,
                        metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--loss-type', default='bce', type=str,
                        help='The loss types: xentropy | bce | centerloss | ...')
    parser.add_argument('--optimizer-type', default='adamw', type=str,
                        help='sgd | adamw | ...')
    
    # model loading and saving
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--pth-save-dir', default='/tmp', type=str,
                        help='The folder to save pths')
    parser.add_argument('--log-save-dir', default='/tmp/logs', type=str,
                        help='The folder to save training logs')
    parser.add_argument('--pth-save-iter', default=1, type=int,
                        help='The iteration to save pth')

    
    # distributed training
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--world-size', default=2, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    args = parser.parse_args()
    # if args.train_config_path != "" and os.path.exists(args.train_config_path):
    #     with open(args.train_config_path) as train_config:
    #         for key, value in yaml.load(train_config).items():
    #             value = type(getattr(args, key))(value) if hasattr(args, key) and getattr(args, key) is not None else value
    #             setattr(args, key, value)
    return args