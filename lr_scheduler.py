import torch
from torch.optim.lr_scheduler import _LRScheduler

class WarmupDecayLR(_LRScheduler):
    '''The schedular increase learning rate linearly from lr_0 to lr_max before epoch T_0, 
    and decays learning rate by multiplying alpha for every T_decay epochs.'''
    
    def __init__(self, optimizer, T_0, T_decay, lr_0, lr_max, alpha, last_epoch=-1):
        self.T_0 = T_0
        self.T_decay = T_decay
        self.lr_0 = lr_0
        self.lr_max = lr_max
        self.alpha = alpha
        super(WarmupDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_0:
            return [self.lr_0 + (self.lr_max - self.lr_0) * (self.last_epoch / self.T_0) for base_lr in self.base_lrs]
        else:
            return [base_lr * (self.alpha ** ((self.last_epoch - self.T_0) // self.T_decay)) for base_lr in self.base_lrs]
        