method = 'PredNet'
# model
stack_sizes = (1, 32, 64, 128, 256) # 1 refer to num of channel(input)
R_stack_sizes = stack_sizes
A_filt_sizes = (3, 3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3, 3)
pixel_max = 1.0
weight_mode = 'L_0'
error_activation = 'relu'
A_activation = 'relu'
LSTM_activation = 'tanh'
LSTM_inner_activation = 'hard_sigmoid'
# training
# lr = 1e-3
batch_size = 16
sched = 'onecycle'