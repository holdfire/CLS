import torch
import torch.nn as nn
from termcolor import cprint


def build_loss(args):
    if args.loss_type == "bce":
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        cprint('WARN => in train.py: loss function is {}'.format(args.loss_type), "yellow")
    elif args.loss_type == "lsce":
        criterion = LabelSmoothCE().cuda(args.gpu)
        cprint('WARN => in train.py: loss function is {}'.format(args.loss_type), "yellow")
    else:
        raise Exception("loss type is not defined.")
    return criterion


class LabelSmoothCE(nn.Module):
    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothCE, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        logits = logits.float() 
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * lb_one_hot, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss



if __name__ == "__main__":
    # test code
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss-type', default='bce')
    args = parser.parse_args()
    criterion = build_loss(args)