import sys
import os
from collections import defaultdict
import torch
import numpy as np
import shutil

def schedule_lr(optimizer, step, total_steps, peak_lr, min_lr, warmup_steps):
    """Linear warmup to peak_lr, then cosine to min_lr; sets the optimizer's lr and returns it."""
    if warmup_steps > 0 and step < warmup_steps:
        lr = peak_lr * (step + 1) / warmup_steps
    else:
        import math
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = min_lr + (peak_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class Logger(object):
    def __init__(self, saver_dir, saver_name_pre):
        self.terminal = sys.stdout
        output_file = os.path.join(saver_dir, saver_name_pre+"-record.log") 
        self.log = open(output_file, "w")

    def write(self, message):
        print(message, end="", file=self.terminal, flush=True)
        print(message, end="", file=self.log, flush=True)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

class Pack(dict):
    def __getattr__(self, name):
        return self[name]

    def add(self, kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def copy(self):
        pack = Pack()
        for k, v in self.items():
            if type(v) is list:
                pack[k] = list(v)
            else:
                pack[k] = v
        return pack

class LossManager(object):
    def __init__(self):
        self.losses = defaultdict(list)
        self.backward_losses = []

    def add_loss(self, loss):
        for key, val in loss.items():
            if val is not None and type(val) is not bool:
                try:
                    self.losses[key].append(val.item())
                except:
                    self.losses[key].append(val)
                    
    def add_backward_loss(self, loss):
        self.backward_losses.append(loss.item())

    def clear(self):
        self.losses = defaultdict(list)
        self.backward_losses = []

    def pprint(self, window=None, prefix=None):
        str_losses = []
        for key, loss in self.losses.items():
            if loss is None:
                continue
            else:
                avg_loss = np.average(loss) if window is None else np.average(loss[-window:])
                str_losses.append("{} {:.4f},".format(key, avg_loss))
        if prefix:
            return "{} {}".format(prefix, " ".join(str_losses))
        else:
            return "{}".format(" ".join(str_losses))

    def avg_loss(self):
        return np.mean(self.backward_losses) 

