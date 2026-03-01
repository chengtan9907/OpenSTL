# ===========================================================================
# OpenSTL SimVP Moving MNIST Base Configuration
# ===========================================================================
# A reference configuration for SimVP on Moving MNIST dataset.
# This can be used as a starting point for your experiments.
# ===========================================================================

method = 'SimVP'

# ---------------------------------------------------------------------------
# Model Parameters
# ---------------------------------------------------------------------------
spatio_kernel_enc = 3
spatio_kernel_dec = 3

# Architecture variant
# Option 1: Standard gSTA
hid_S = 64
hid_T = 512
N_T = 8
N_S = 4
model_type = 'gSTA'

# Option 2: Light version (uncomment to use)
# hid_S = 32
# hid_T = 256
# N_T = 4
# N_S = 2
# model_type = 'gSTA'

# Option 3: With different backbones (uncomment one to use)
# model_type = 'ViT'        # Vision Transformer
# model_type = 'Swin'       # Swin Transformer
# model_type = 'ConvNeXt'   # ConvNeXt
# model_type = 'MLPMixer'   # MLP-Mixer
# model_type = 'VAN'        # Vision Attention Network
# model_type = 'HorNet'     # HorNet
# model_type = 'Uniformer'  # Uniformer
# model_type = 'MogaNet'    # MogaNet

# Regularization
drop_path = 0.1

# ---------------------------------------------------------------------------
# Training Parameters
# ---------------------------------------------------------------------------
lr = 1e-3
batch_size = 16
epoch = 100

# Scheduler: 'onecycle' works well for Moving MNIST
sched = 'onecycle'
warmup_lr = 1e-6
min_lr = 1e-5
warmup_epoch = 10

# Optimizer
opt = 'adam'
weight_decay = 0.0

# ---------------------------------------------------------------------------
# Data Parameters
# ---------------------------------------------------------------------------
dataname = 'moving_mnist'
in_shape = (10, 1, 64, 64)  # (pre_seq_length, channels, height, width)
total_length = 20
aft_seq_length = 10

# ---------------------------------------------------------------------------
# Experiment Settings
# ---------------------------------------------------------------------------
seed = 42
gpus = [0]
dist = 0
log_step = 10
metric_for_bestckpt = 'val_loss'
ex_name = 'simvp_mmnist_baseline'
