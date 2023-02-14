from .optim_scheduler import get_optim_scheduler


class Base_method(object):
    def __init__(self, args, device, steps_per_epoch):
        super(Base_method, self).__init__()
        self.args = args
        self.device = device
        self.config = args.__dict__
        self.criterion = None

    def _build_model(self, **kwargs):
        raise NotImplementedError

    def _init_optimizer(self, steps_per_epoch):
        return get_optim_scheduler(self.args, self.args.epoch, self.model, steps_per_epoch)

    def train_one_epoch(self, train_loader, **kwargs): 
        '''
        Train the model with train_loader.
        Input params:
            train_loader: dataloader of train.
        '''
        raise NotImplementedError

    def vali_one_epoch(self, vali_loader, **kwargs):
        '''
        Evaluate the model with val_loader.
        Input params:
            val_loader: dataloader of validation.
        '''
        raise NotImplementedError

    def test_one_epoch(self, test_loader, **kwargs):
        raise NotImplementedError