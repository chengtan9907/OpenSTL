method = 'MMVP'
# model
downsample_setting = '2,2,2'
hid_S = 16
hid_T = 96
rrdb_encoder_num = 2
rrdb_decoder_num = 2
rrdb_enhance_num = 2
use_direct_predictor = True
# training
lr = 1e-3
batch_size = 16
sched = 'onecycle'