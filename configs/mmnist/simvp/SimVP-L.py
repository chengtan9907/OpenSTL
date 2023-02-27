method = 'SimVP'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
# model_type = None  # define `model_type` in args
hid_S = 128
hid_T = 1024
N_T = 24
N_S = 4
# training
lr = 1e-3
batch_size = 16
drop_path = 0
sched = 'onecycle'