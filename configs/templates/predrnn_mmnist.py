# ===========================================================================
# OpenSTL PredRNN Moving MNIST Base Configuration
# ===========================================================================
# A reference configuration for PredRNN on Moving MNIST dataset.
# ===========================================================================

method = 'PredRNN'

# ---------------------------------------------------------------------------
# Model Parameters
# ---------------------------------------------------------------------------
# Number of layers
num_layers = 4

# Hidden state dimensions
hidden_dim = 128

# LSTM cell parameters
filter_size = 5
stride = 1
padding = 2

# Memory compression (for PredRNN++)
# memory_decimation = 2

# ---------------------------------------------------------------------------
# Training Parameters
# ---------------------------------------------------------------------------
lr = 1e-3
batch_size = 8
epoch = 100

# Scheduler
sched = 'cosine'
warmup_lr = 1e-6
min_lr = 1e-5
warmup_epoch = 10

# Optimizer
opt = 'adam'
weight_decay = 1e-4

# Gradient clipping
grad_clip = 5.0

# ---------------------------------------------------------------------------
# Data Parameters
# ---------------------------------------------------------------------------
dataname = 'moving_mnist'
in_shape = (10, 1, 64, 64)
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
ex_name = 'predrnn_mmnist_baseline'
