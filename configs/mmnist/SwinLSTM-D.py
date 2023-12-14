method = 'SwinLSTM_D'
# model
depths_downsample = '2,6'
depths_upsample = '6,2'
num_heads = '4,8'
patch_size = 2
window_size = 4
embed_dim = 128
# training
lr = 1e-4
batch_size = 16
sched = 'onecycle'