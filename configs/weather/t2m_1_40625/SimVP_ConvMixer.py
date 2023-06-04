method = 'SimVP'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'convmixer'
hid_S = 32
hid_T = 256
N_T = 8
N_S = 2
# training
lr = 1e-2
batch_size = 4 # bs16 = 4gpus x bs4
drop_path = 0.1
sched = 'cosine'
warmup_epoch = 0
