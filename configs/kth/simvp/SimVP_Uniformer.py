method = 'SimVP'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'uniformer'
hid_S = 64
hid_T = 256
N_T = 4
N_S = 2
# training
lr = 1e-3
drop_path = 0.1
batch_size = 2  # bs = 2 x 4GPUs
sched = 'onecycle'