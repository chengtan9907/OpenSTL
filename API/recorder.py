import numpy as np
import torch

class Recorder:
    def __init__(self, verbose=False, delta=0):
        self.verbose = verbose
        self.best_score = None
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score >= self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss