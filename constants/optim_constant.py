optim_parameters = {
    'opt': 'adamw',
    'opt_eps': None,
    'opt_betas': None,
    'momentum': 0.9,
    'weight_decay': 0.05,
    'clip_grad': None,
    'clip_mode': 'norm'
}
schedule_parameters = {
    'sched': 'cosine',
    'lr_noise': None,
    'lr_noise_pct': 0.67,
    'lr_noise_std': 1.0,
    'lr_cycle_mul': 1.0,
    'lr_cycle_decay': 0.5,
    'lr_cycle_limit': 1,
    'lr_k_decay': 1.0,
    'warmup_lr': 1e-6,
    'min_lr': 1e-6,
    'epoch_repeats': 0.,
    'start_epoch': None,
    'decay_epoch': 100,
    'warmup_epochs': 5,
    'cooldown_epochs': 10,
    'patience_epochs': 10,
    'decay_rate': 0.1
}