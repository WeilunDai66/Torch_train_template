import os
import random
import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

import numpy as np


class CosineAnnealingWarmupLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, warmup_iters, eta_min=0, mode='linear', last_epoch=-1):
        self.mode = mode
        self.warmup_iter = warmup_iters
        self.eta_min = eta_min
        self.total_iters = total_iters

        super(CosineAnnealingWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # if self.last_epoch >= 30 * 226:
        #     self.last_epoch = 29 * 226

        if self.mode == 'exp':
            w = min(1, 1 - (math.exp(-(self.last_epoch) / self.warmup_iter)))
        elif self.mode == 'linear':
            w = min(1, self.last_epoch / self.warmup_iter)

        return [w * (self.eta_min + (base_lr - self.eta_min) *
                     (1 + math.cos(math.pi * self.last_epoch / self.total_iters)) / 2)
                for base_lr in self.base_lrs]



def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False