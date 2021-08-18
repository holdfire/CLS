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
# from termcolor import cprint
import warnings
import tensorboard
from loguru import logger
import tqdm
from timm.utils import CheckpointSaver

import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torch.distributed as dist
import torch.multiprocessing as mp

from models import build_model
from dataloader import build_dataloader
from loss import build_loss
from lr_schedule import build_lr_schedule
from optimizer import build_optimizer

import parser
from tools.utils import AverageMeter, ProgressMeter, accuracy, cal_metrics, find_best_threshold


def main(args):

    args = parser.parse_args()
    # add logs
    if not os.path.exists(args.log_save_dir):
        os.makedirs(args.log_save_dir)
    train_log = os.path.join(args.log_save_dir, 'train_log')
    logger.add(train_log)
    args.logger = logger

    ##################### When debuging, annoting this ###################
    # if args.gpu != 0:
    #     def print_pass(*args):
    #         pass
    #     builtins.print = print_pass
    #######################################################################
    
    # set random seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    print('running on GPU {}'.format(args.local_rank))

    # Distributed Date Parallel
    dist.init_process_group(backend='nccl', init_method="env://")
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    # create dataloader
    args.mode = "train"
    train_loader, train_sampler = build_dataloader(args)
    args.n_iter_per_epoch = len(train_loader)
    if args.evaluate:
        args.mode = "val"
        val_loader, val_sampler = build_dataloader(args)
    

    # create model
    model = build_model.build_model(args).to(device)
    args.model = model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    criterion = build_loss(args)
    optimizer = build_optimizer(args)
    scheduler = build_lr_schedule(args, optimizer)

    # saving checkpoint by package timm        
    saver = None
    if args.local_rank == 0:
        saver = CheckpointSaver(model, optimizer, args=args, decreasing=True, checkpoint_dir=args.pth_save_dir, recovery_dir=args.pth_save_dir, max_history=args.epochs+1)
    
    total_epoch = len(train_loader) // args.world_size
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        train(args, train_loader, model, criterion, optimizer, scheduler, epoch, total_epoch, logger)
        if args.evaluate and epoch % args.eval_interval_time == 0:
            validate(args, val_loader, model, criterion, scheduler, saver, epoch, total_epoch, logger)



def train(args, train_loader, model, criterion, optimizer, scheduler, epoch, total_epoch, logger):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    learning_rate = AverageMeter('LR:', ':.4e')
    model.train()

    for i, (images, target) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

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

        # new print strategy
        if args.local_rank == 0:
            if i % args.print_freq == 0:
                info = 'epoch {} step {} LR: {} train_acc: {:.5f}({:.5f}) train_loss: {:.5f}({:.5f})'.format(
                    epoch, i, learning_rate.val, top1.val, top1.avg, losses.val, losses.avg)
                logger.info(info)


def validate(args, val_loader, model, criterion, scheduler, saver, epoch, total_epoch, logger):
    loss_m = AverageMeter('Acc@1', ':6.2f')
    model.eval()

    y_preds, y_targets = [], []   
    for _, datas in enumerate(val_loader):
        with torch.no_grad():
            images = datas[0].cuda(non_blocking=True)
            targets = datas[1].cuda(non_blocking=True)
            bs = targets.size(0)

            embedding, output = model(images)

            loss = criterion(output, targets)
            loss_m.update(loss.item(), bs)

            probs = torch.softmax(output, dim=1)[:,1]
            y_preds.extend(probs)
            y_targets.extend(targets)

    y_preds, y_targets = torch.stack(y_preds), torch.stack(y_targets)
    world_size = dist.get_world_size()

    gather_y_preds = [torch.ones_like(y_preds) for _ in range(world_size)]
    dist.all_gather(gather_y_preds, y_preds)
    gather_y_preds = torch.cat(gather_y_preds).cpu().tolist()

    gather_y_targets = [torch.ones_like(y_targets) for _ in range(world_size)]
    dist.all_gather(gather_y_targets, y_targets)
    gather_y_targets = torch.cat(gather_y_targets).cpu().tolist()

    metrics = cal_metrics(gather_y_targets, gather_y_preds, threshold='auto')
    scheduler.step(metrics.ACER)

    if args.local_rank == 0:
        # revised saving strategy
        save_state = {
            'epoch': epoch,
            # 'arch': type(model).__name__.lower(),
            'state_dict': model.state_dict(),
            # 'optimizer': saver.optimizer.state_dict(),
            # 'version': 2,  # version < 2 increments epoch before save
        }
        filename = '-'.join([saver.save_prefix, str(epoch)]) + saver.extension
        save_path = os.path.join(saver.checkpoint_dir, filename)
        torch.save(save_state, save_path)
        # print("here")

        # old saving strategy
        # best_metric, best_epoch = saver.save_checkpoint(epoch)

        for k, v in metrics.items():
            args.logger.info('val_{}: {:.4f}'.format(k, v * 100))

        cur_lr = [group['lr'] for group in scheduler.optimizer.param_groups][0]
        info = 'VAL_INFO EPOCH {}:\n\tTPR: {:.4f} FPR: {:.4f} AUC: {:.4f} ACC: {:.4f} Loss: {:.4f} lr: {:.5f}'.format(
            epoch, metrics.TPR, metrics.FPR, metrics.AUC, metrics.ACC, loss_m.avg, cur_lr
        )
        args.logger.info(info)
        # args.logger.info('best_epoch: {} best_val_ACER: {:.4f}'.format(best_epoch, best_metric * 100))



if __name__ == '__main__':
    # cprint('WARN => torch version : {}, torchvision version : {}'.format(torch.__version__, torchvision.__version__), color='yellow')
    args = parser.parse_args()
    pprint.pprint(vars(args))
    if not os.path.exists(args.pth_save_dir):
        os.makedirs(args.pth_save_dir)
    # if not os.path.exists(args.log_save_dir):
    #     os.makedirs(args.log_save_dir)
    main(args)
