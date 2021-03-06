"""
Reference: 
https://github.com/microsoft/Swin-Transformer/blob/main/lr_scheduler.py
https://github.com/rwightman/pytorch-image-models/tree/master/timm/scheduler
"""

import torch
import timm
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler


def build_lr_schedule(args, optimizer):
    num_steps = int(args.epochs * args.n_iter_per_epoch)
    # warmup_steps = int(args.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
    # decay_steps = int(args.TRAIN.LR_SCHEDULER.DECAY_EPOCHS * n_iter_per_epoch)

    lr_scheduler = None
    if args.lr_schedule == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            t_mul=1.,
            lr_min=5e-6,
            # warmup_lr_init=args.TRAIN.WARMUP_LR,
            # warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    elif args.lr_schedule == 'linear':
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=0.01,
            # warmup_lr_init=args.TRAIN.WARMUP_LR,
            # warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif args.lr_schedule == 'reducePlateau':
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=10, verbose=True)
    # to do
    elif args.lr_schedule == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=args.TRAIN.LR_SCHEDULER.DECAY_RATE,
            t_in_epochs=False,
        )

    else:
        raise Exception("lr_schedule is not defined.")
    
    return lr_scheduler


class LinearLRScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min_rate: float,
                 warmup_t=0,
                 warmup_lr_init=0.,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None



if __name__ == "__main__":

    # test code
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_schedule', default='cosine')
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--lr', default=0.01)
    parser.add_argument('--weight-decay', default=1e-4)
    args = parser.parse_args()

    lr_schedule = build_lr_schedule(args)
