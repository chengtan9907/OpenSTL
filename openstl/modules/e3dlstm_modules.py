import torch
import torch.nn as nn
import torch.nn.functional as F


class tf_Conv3d(nn.Module):

    def __init__(self, in_channels, out_channels, *vargs, **kwargs):
        super(tf_Conv3d, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, *vargs, **kwargs)

    def forward(self, input):
        return F.interpolate(self.conv3d(input), size=input.shape[-3:], mode="nearest")


class Eidetic3DLSTMCell(nn.Module):

    def __init__(self, in_channel, num_hidden, window_length,
                 height, width, filter_size, stride, layer_norm):
        super(Eidetic3DLSTMCell, self).__init__()

        self._norm_c_t = nn.LayerNorm([num_hidden, window_length, height, width])
        self.num_hidden = num_hidden
        self.padding = (0, filter_size[1] // 2, filter_size[2] // 2) 
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                tf_Conv3d(in_channel, num_hidden * 7, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 7, window_length, height, width])
            )
            self.conv_h = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, window_length, height, width])
            )
            self.conv_gm = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, window_length, height, width])
            )
            self.conv_new_cell = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, window_length, height, width])
            )
            self.conv_new_gm = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, window_length, height, width])
            )
        else:
            self.conv_x = nn.Sequential(
                tf_Conv3d(in_channel, num_hidden * 7, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_gm = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_new_cell = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_new_gm = nn.Sequential(
                tf_Conv3d(num_hidden, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = tf_Conv3d(num_hidden * 2, num_hidden, kernel_size=1,
                                   stride=1, padding=0, bias=False)
    
    def _attn(self, in_query, in_keys, in_values):
        batch, num_channels, _, width, height = in_query.shape
        query = in_query.reshape(batch, -1, num_channels)
        keys = in_keys.reshape(batch, -1, num_channels)
        values = in_values.reshape(batch, -1, num_channels)
        attn = torch.einsum('bxc,byc->bxy', query, keys)
        attn = torch.softmax(attn, dim=2)
        attn = torch.einsum("bxy,byc->bxc", attn, values)
        return attn.reshape(batch, num_channels, -1, width, height)

    def forward(self, x_t, h_t, c_t, global_memory, eidetic_cell):
        h_concat = self.conv_h(h_t)
        i_h, g_h, r_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        x_concat = self.conv_x(x_t)
        i_x, g_x, r_x, o_x, temp_i_x, temp_g_x, temp_f_x = \
            torch.split(x_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        r_t = torch.sigmoid(r_x + r_h)
        g_t = torch.tanh(g_x + g_h)

        new_cell = c_t + self._attn(r_t, eidetic_cell, eidetic_cell)
        new_cell = self._norm_c_t(new_cell) + i_t * g_t

        new_global_memory = self.conv_gm(global_memory)
        i_m, f_m, g_m, m_m = torch.split(new_global_memory, self.num_hidden, dim=1)

        temp_i_t = torch.sigmoid(temp_i_x + i_m)
        temp_f_t = torch.sigmoid(temp_f_x + f_m + self._forget_bias)
        temp_g_t = torch.tanh(temp_g_x + g_m)
        new_global_memory = temp_f_t * torch.tanh(m_m) + temp_i_t * temp_g_t
        
        o_c = self.conv_new_cell(new_cell)
        o_m = self.conv_new_gm(new_global_memory)

        output_gate = torch.tanh(o_x + o_h + o_c + o_m)

        memory = torch.cat((new_cell, new_global_memory), 1)
        memory = self.conv_last(memory)

        output = torch.tanh(memory) * torch.sigmoid(output_gate)

        return output, new_cell, global_memory
