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
lr = 5e-3
batch_size = 16
drop_path = 0.2
sched = 'cosine'
warmup_epoch = 0