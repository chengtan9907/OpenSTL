method = 'MAU'
# scheduled sampling
scheduled_sampling = 1
sampling_stop_iter = 50000
sampling_start_value = 1.0
sampling_changing_rate = 0.00002
# model
num_hidden = '64,64,64,64'
filter_size = 5
stride = 1
patch_size = 1
layer_norm = 0
sr_size = 4
tau = 5
cell_mode = 'normal'
model_mode = 'normal'
# training
lr = 5e-4
batch_size = 16
sched = 'onecycle'
# noisy type
noise_type = 'missing'