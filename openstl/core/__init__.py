# Copyright (c) CAIRI AI Lab. All rights reserved

from .metrics import metric
from .optim_scheduler import get_optim_scheduler, timm_schedulers
from .optim_constant import optim_parameters
from .recorder import Recorder
from .ema_hook import EMAHook, SwitchEMAHook
from .hooks import Hook, Priority, get_priority

hook_maps = {
    'emahook': EMAHook,
    **dict.fromkeys(['semahook', 'switchemahook'], SwitchEMAHook),
}

__all__ = [
    'Hook', 'EMAHook', 'SwitchEMAHook', 'Priority', 'Recorder'
    'metric', 'get_optim_scheduler', 'optim_parameters', 'timm_schedulers'
]