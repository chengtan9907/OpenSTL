method = 'SimVP'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'convmixer'
hid_S = 64
hid_T = 256
N_T = 6
N_S = 2
# training
lr = 1e-2
drop_path = 0.1
batch_size = 8  # bs = 8 x 2GPUs
sched = 'onecycle'