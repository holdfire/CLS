import torch
import torch.nn as nn
from termcolor import cprint


def build_loss(args):
    if args.loss_type == "bce":
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        cprint('WARN => in train.py: loss function is {}'.format(args.loss_type), "yellow")
    else:
        raise Exception("loss type is not defined.")
    
    return criterion


if __name__ == "__main__":
    # test code
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss-type', default='bce')
    args = parser.parse_args()
    criterion = build_loss(args)