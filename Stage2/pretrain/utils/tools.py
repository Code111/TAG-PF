import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from typing import Optional
plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience: int = 3, verbose: bool = True):
        self.patience = int(patience)
        self.verbose = bool(verbose)
        self.counter = 0
        self.best: Optional[float] = None
        self.early_stop = False

    def step(self, val_loss: float) -> bool:
        improved = (self.best is None) or (val_loss < self.best)
        if improved:
            if self.verbose:
                print(f"Validation loss improved ({self.best} -> {val_loss:.6f}).")
            self.best = float(val_loss)
            self.counter = 0
            return True

        self.counter += 1
        if self.verbose:
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
        if self.counter >= self.patience:
            self.early_stop = True
        return False
    


    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



def visual(true, preds=None, name='./pic/test.pdf'):

    plt.figure(figsize=(8, 5))

    plt.plot(true, label='GroundTruth', linewidth=2.2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=1.8, alpha=0.9)

    plt.autoscale(axis='y')

    ymin, ymax = plt.gca().get_ylim()
    upper_margin = 0.06 * (ymax - ymin)
    lower_margin = 0.01 * (ymax - ymin)   
    plt.ylim(ymin - lower_margin, ymax + upper_margin)

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='best', frameon=False)

    plt.xlabel('Time Step')
    plt.ylabel('Value')

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(name, bbox_inches='tight')
    plt.close()

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


