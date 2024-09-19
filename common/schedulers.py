import torch
import torch.optim as optim

def WarmupLR(optimizer, warmup_step=0,  down_step=5e4, max_lr=1e-4, min_lr=1e-5, **kwargs):
    alpha = (max_lr - 1e-5) / warmup_step**2
    def lr_lambda(step):
        init_lr = 1e-5
        s1, s2 = warmup_step, warmup_step + down_step
        if step < s1:
            return init_lr + alpha * step**2
        elif s1 <= step < s2:
            return (max_lr - min_lr) / (s1 - s2) * step + (min_lr*s1 - max_lr*s2) / (s1 - s2)
        else:
            return min_lr
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
