import torch
import torch.nn as nn

from openstl.modules import Eidetic3DLSTMCell


class E3DLSTM_Model(nn.Module):
    r"""E3D-LSTM Model

    Implementation of `EEidetic 3D LSTM: A Model for Video Prediction and Beyond
    <https://openreview.net/forum?id=B1lKS2AqtX>`_.

    """

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(E3DLSTM_Model, self).__init__()
        T, C, H, W = configs.in_shape

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        self.window_length = 2
        self.window_stride = 1

        height = H // configs.patch_size
        width = W // configs.patch_size
        self.MSE_criterion = nn.MSELoss()
        self.L1_criterion = nn.L1Loss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                Eidetic3DLSTMCell(in_channel, num_hidden[i],
                                  self.window_length, height, width, (2, 5, 5),
                                  configs.stride, configs.layer_norm))
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv3d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=(self.window_length, 1, 1),
                                   stride=(self.window_length, 1, 1), padding=0, bias=False)

    def forward(self, frames_tensor, mask_true, **kwargs):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        device = frames_tensor.device
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        c_history = []
        input_list = []

        for t in range(self.window_length - 1):
            input_list.append(
                torch.zeros_like(frames[:, 0]))

        for i in range(self.num_layers):
            zeros = torch.zeros(
                [batch, self.num_hidden[i], self.window_length, height, width], device=device)
            h_t.append(zeros)
            c_t.append(zeros)
            c_history.append(zeros)

        memory = torch.zeros(
            [batch, self.num_hidden[0], self.window_length, height, width], device=device)

        for t in range(self.configs.pre_seq_length + self.configs.aft_seq_length - 1):
            # reverse schedule sampling
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.configs.pre_seq_length:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.configs.pre_seq_length] * frames[:, t] + \
                          (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen

            input_list.append(net)

            if t % (self.window_length - self.window_stride) == 0:
                net = torch.stack(input_list[t:], dim=0)
                net = net.permute(1, 2, 0, 3, 4).contiguous()

            for i in range(self.num_layers):
                if t == 0:
                    c_history[i] = c_t[i]
                else:
                    c_history[i] = torch.cat((c_history[i], c_t[i]), 1)
                
                input = net if i == 0 else h_t[i-1]
                h_t[i], c_t[i], memory = self.cell_list[i](input, h_t[i], c_t[i], memory, c_history[i])

            x_gen = self.conv_last(h_t[self.num_layers - 1]).squeeze(2)
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:]) + \
                self.L1_criterion(next_frames, frames_tensor[:, 1:])
        else:
            loss = None

        return next_frames, loss
