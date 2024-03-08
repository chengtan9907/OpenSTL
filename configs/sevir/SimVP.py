method = 'SimVP'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'IncepU'
hid_S = 64
hid_T = 256
N_T = 4
N_S = 2
# training
lr = 5e-3
batch_size = 8
drop_path = 0.1
sched = 'onecycle'
warmup_epoch = 0
metric_threshold = 74