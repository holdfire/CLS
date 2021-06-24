#!/usr/bin/python3
"""
Author: Liu Xing
Date: 2021-05-13
Reference: https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""

import os
import time
import pprint
import builtins
from termcolor import cprint
import warnings
import tensorboard
warnings.filterwarnings("ignore")

import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from models import build_model
from dataloader import build_train_loader
from loss import build_loss
from lr_schedule import build_lr_schedule
from optimizer import build_optimizer

import parser
from tools.utils import AverageMeter, ProgressMeter, accuracy, save_checkpoint


def main(args):
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        cprint("WARN => in train.py: Use GPU: {} for training".format(args.gpu), 'yellow')

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    # create tensorboard
    writer = SummaryWriter(log_dir = args.log_save_dir, max_queue=50, flush_secs=120)

    # create model
    model = build_model.build_model(args)
    args.model = model
    cprint(" Model => ")
    print(model)

    if not torch.cuda.is_available():
        cprint('WARN => in train.py: using CPU, this will be slow', "red")
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()


    # build dataloader
    train_loader, train_sampler = build_train_loader(args)


    # training on ImageNet: Data loading code
    # args.data = "/home/data4/ILSVRC2012"
    # print("training on imageNet")
    # traindir = os.path.join(args.data, 'train')
    # valdir = os.path.join(args.data, 'val')
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))
    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # else:
    #     train_sampler = None
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    #     num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    

    args.n_iter_per_epoch = len(train_loader)

    # build loss function (criterion), optimizer and learning rate schedule
    criterion = build_loss(args)
    optimizer = build_optimizer(args)
    scheduler = build_lr_schedule(args, optimizer)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            cprint("WARN => loading checkpoint: '{}'".format(args.resume), 'yellow')
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)

            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            cprint("WARN => loaded checkpoint: '{}' (epoch {})".format(args.resume, checkpoint['epoch']), 'yellow')
        else:
            cprint("WARN => no checkpoint found at: '{}'".format(args.resume), 'red')

    cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, scheduler, writer)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                # 'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, os.path.join(args.pth_save_dir, "epoch_" + str(epoch+1) + ".pth.tar"))

    writer.close()


def train(train_loader, model, criterion, optimizer, epoch, args, scheduler, writer):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    learning_rate = AverageMeter('LR:', ':.4e')

    progress = ProgressMeter(
        len(train_loader),
        [losses, top1, learning_rate],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        embedding, output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1, ))[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iters = epoch * args.n_iter_per_epoch + i
        scheduler.step_update(iters)
        learning_rate.update(scheduler.get_update_values(iters)[0])

        # save checkpoint by iteration
        assert args.pth_save_iter <= len(train_loader), \
               'please make args.pth_save_iter smalller than total iterations'
        
        if args.pth_save_iter != 0 and i % args.pth_save_iter == 0 and i != 0 :
            fn = os.path.join(
                    args.pth_save_dir, 'iter_{}.pth.tar'.format(
                    str(iters).zfill(12)))
            save_checkpoint({
                # 'epoch': epoch + 1,
                # 'arch': args.arch,
                'state_dict': model.state_dict(),
                # 'optimizer' : optimizer.state_dict(),
                # 'scheduler': scheduler.state_dict(),
            }, filename=fn)
            cprint('Saving by iteration => save pth for epoch {} at {}'.format(epoch + 1, fn))


        # display frequency and tensorboard
        writer.add_scalars('training',
                          {'loss': losses.val, 
                           'top1': top1.val,
                           'loss_avg': losses.avg, 
                           'top1_avg': top1.avg},
                           global_step = iters,)

        if i % args.print_freq == 0:
            progress.display(i)



if __name__ == '__main__':
    cprint('WARN => torch version : {}, torchvision version : {}'.format(torch.__version__, torchvision.__version__), color='yellow')
    args = parser.parse_args()
    pprint.pprint(vars(args))
    if not os.path.exists(args.pth_save_dir):
        os.makedirs(args.pth_save_dir)
    # if not os.path.exists(args.log_save_dir):
    #     os.makedirs(args.log_save_dir)
    main(args)
