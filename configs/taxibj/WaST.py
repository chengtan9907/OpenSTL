method = 'WaST'
# model
encoder_dim = 20
block_list = [2, 8, 2]
mlp_ratio = 4.0
loss_weight = 0.001
# drop scheduler
cutmode = "late"  # early or late
cutoff = 20
# training
lr = 1e-3  # 5e-3
batch_size = 16
drop_path = 0.25
sched = 'cosine'
warmup_epoch = 0

# CUDA_VISIBLE_DEVICES=3 python tools/train.py -d taxibj --epoch 50 -c configs/taxibj/WaST.py --ex_name taxibj_wast_ep50
