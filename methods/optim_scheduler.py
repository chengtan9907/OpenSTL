

def get_optim_scheduler(lr, epoch, model, steps_per_epoch):
    from torch import optim
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epoch * steps_per_epoch)
    return optimizer, scheduler