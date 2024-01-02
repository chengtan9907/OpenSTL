from .simvp import SimVP
from openstl.modules.wast_modules import *
from openstl.core.drop_scheduler import drop_scheduler

class WaST(SimVP):
    def __init__(self, **args):
        SimVP.__init__(self, **args)
        self.steps_per_epoch = args["steps_per_epoch"]
        self.drop_scheduler = drop_scheduler(
            drop_rate=args["drop_path"], epochs=args["epoch"], niter_per_ep=self.steps_per_epoch,
            cutoff_epoch=args["cutoff"], mode=args["cutmode"], schedule="constant") if "cutoff" in args else None
        self.hffl = HighFocalFrequencyLoss(loss_weight=args["loss_weight"]) if "loss_weight" in args else None

    def _build_model(self, **args):
        return WaST_level1(**args).to(self.device)
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x)
        loss = self.criterion(pred_y, batch_y)
        loss = loss + self.hffl(pred_y, batch_y, reshape=True) if self.hffl is not None else loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        if hasattr(self, 'drop_scheduler'):
            steps = self.current_epoch * self.steps_per_epoch + batch_idx
            self.model.update_drop_path(drop_path_rate=self.drop_scheduler[steps])
        return loss