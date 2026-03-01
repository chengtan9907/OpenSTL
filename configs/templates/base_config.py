# ===========================================================================
# OpenSTL Base Configuration Template
# ===========================================================================
# This is a base configuration template for OpenSTL experiments.
# Copy this file and modify the parameters for your specific experiment.
# ===========================================================================

# ---------------------------------------------------------------------------
# Method Selection
# ---------------------------------------------------------------------------
# Available methods: simvp, convlstm, predrnn, predrnnv2, predrnnpp,
#                    mau, mim, phydnet, e3dlstm, tau, mmvp,
#                    swinlstm_d, swinlstm_b, wast
method = 'SimVP'

# ---------------------------------------------------------------------------
# Model Architecture Parameters
# ---------------------------------------------------------------------------
# These parameters control the model architecture.
# Adjust based on your method and dataset.

# Spatial convolution kernel size (encoder/decoder)
spatio_kernel_enc = 3
spatio_kernel_dec = 3

# Hidden dimensions
hid_S = 64      # Spatial hidden dimension
hid_T = 512     # Temporal hidden dimension

# Number of layers/blocks
N_T = 8         # Number of temporal blocks
N_S = 4         # Number of spatial blocks

# Model type for SimVP (optional):
# None, 'gSTA', 'ViT', 'Swin', 'ConvNeXt', 'MLPMixer', 'VAN', 'HorNet', etc.
# model_type = 'gSTA'

# Dropout / DropPath rate
drop_path = 0.1

# ---------------------------------------------------------------------------
# Training Hyperparameters
# ---------------------------------------------------------------------------
# Learning rate
lr = 1e-3

# Batch size (per GPU)
batch_size = 16

# Number of epochs
epoch = 100

# Learning rate scheduler
# Available: 'onecycle', 'cosine', 'step', 'multistep', 'tanh'
sched = 'onecycle'

# Optimizer
# Available: 'adam', 'adamw', 'sgd', 'radam', 'adamp', 'adafactor', etc.
opt = 'adam'

# Weight decay
weight_decay = 0.0

# Warmup learning rate
warmup_lr = 1e-6

# Minimum learning rate (for cosine/onecycle scheduler)
min_lr = 1e-5

# Warmup epochs
warmup_epoch = 5

# Decay epoch (for step/multistep scheduler)
decay_epoch = 30

# Decay rate (for step/multistep scheduler)
decay_rate = 0.1

# ---------------------------------------------------------------------------
# Data Parameters
# ---------------------------------------------------------------------------
# Dataset name: moving_mnist, weather, human, taxibj, bair, kinetics, etc.
dataname = 'moving_mnist'

# Input shape: (pre_seq_length, channels, height, width)
# in_shape = (10, 1, 64, 64)

# Total sequence length (input + output)
# total_length = 20

# Prediction sequence length
# aft_seq_length = 10

# Patch size (for methods that use patching)
# patch_size = 4

# ---------------------------------------------------------------------------
# Experiment Settings
# ---------------------------------------------------------------------------
# Random seed
seed = 42

# Device
device = 'cuda'

# GPU IDs to use (list of integers)
# gpus = [0]  # Single GPU
# gpus = [0, 1, 2, 3]  # Multiple GPUs

# Distributed training
dist = 0  # 0: single node, 1: distributed

# Find unused parameters (for DDP)
find_unused_parameters = False

# ---------------------------------------------------------------------------
# Logging & Checkpointing
# ---------------------------------------------------------------------------
# Experiment name (will be saved in work_dirs/)
# ex_name = 'simvp_mmnist_test'

# Result directory
# res_dir = 'work_dirs'

# Log step (frequency of logging)
log_step = 10

# Metric for best checkpoint
metric_for_bestckpt = 'val_loss'

# Save last checkpoint
save_last = True

# Test mode
test = False

# Load from checkpoint
# ckpt_path = 'path/to/checkpoint.ckpt'

# Display method info (model architecture, FLOPs, etc.)
no_display_method_info = False

# Calculate FPS
fps = False

# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------
# Metrics to compute during evaluation
# metrics = ['mse', 'mae', 'ssim', 'psnr']
