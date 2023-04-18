import torch
import torch.nn as nn


class MIMBlock(nn.Module):

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(MIMBlock, self).__init__()

        self.convlstm_c = None
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0

        self.ct_weight = nn.Parameter(torch.zeros(num_hidden*2, height, width))
        self.oc_weight = nn.Parameter(torch.zeros(num_hidden, height, width))

        if layer_norm:
            self.conv_t_cc = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 3, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, height, width])
            )
            self.conv_s_cc = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_x_cc = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_h_concat = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_x_concat = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
        else:
            self.conv_t_cc = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 3, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_s_cc = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_x_cc = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h_concat = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_x_concat = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1,
                                   stride=1, padding=0, bias=False)

    def _init_state(self, inputs):
        return torch.zeros_like(inputs)

    def MIMS(self, x, h_t, c_t):
        if h_t is None:
            h_t = self._init_state(x)
        if c_t is None:
            c_t = self._init_state(x)

        h_concat = self.conv_h_concat(h_t)
        i_h, g_h, f_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        ct_activation = torch.mul(c_t.repeat(1,2,1,1), self.ct_weight)
        i_c, f_c = torch.split(ct_activation, self.num_hidden, dim=1)

        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h

        if x != None:
            x_concat = self.conv_x_concat(x)
            i_x, g_x, f_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)

            i_ = i_ +  i_x
            f_ = f_ + f_x
            g_ = g_ +  g_x
            o_ = o_ + o_x

        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_ + self._forget_bias)
        c_new = f_ * c_t + i_ * torch.tanh(g_)

        o_c = torch.mul(c_new, self.oc_weight)

        h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)

        return h_new, c_new

    def forward(self, x, diff_h, h, c, m):
        h = self._init_state(x) if h is None else h
        c = self._init_state(x) if c is None else c
        m = self._init_state(x) if m is None else m
        diff_h = self._init_state(x) if diff_h is None else diff_h

        t_cc = self.conv_t_cc(h)
        s_cc = self.conv_s_cc(m)
        x_cc = self.conv_x_cc(x)

        i_s, g_s, f_s, o_s = torch.split(s_cc, self.num_hidden, dim=1)
        i_t, g_t, o_t = torch.split(t_cc, self.num_hidden, dim=1)
        i_x, g_x, f_x, o_x = torch.split(x_cc, self.num_hidden, dim=1)

        i = torch.sigmoid(i_x + i_t)
        i_ = torch.sigmoid(i_x + i_s)
        g = torch.tanh(g_x + g_t)
        g_ = torch.tanh(g_x + g_s)
        f_ = torch.sigmoid(f_x + f_s + self._forget_bias)
        o = torch.sigmoid(o_x + o_t + o_s)
        new_m = f_ * m + i_ * g_
        
        c, self.convlstm_c = self.MIMS(diff_h, c, self.convlstm_c \
            if self.convlstm_c is None else self.convlstm_c.detach())

        new_c = c + i * g
        cell = torch.cat((new_c, new_m), 1)
        new_h = o * torch.tanh(self.conv_last(cell))

        return new_h, new_c, new_m


class MIMN(nn.Module):

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(MIMN, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0

        self.ct_weight = nn.Parameter(torch.zeros(num_hidden*2, height, width))
        self.oc_weight = nn.Parameter(torch.zeros(num_hidden, height, width))

        if layer_norm:
            self.conv_h_concat = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_x_concat = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
        else:
            self.conv_h_concat = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_x_concat = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1,
                                   stride=1, padding=0, bias=False)

    def _init_state(self, inputs):
        return torch.zeros_like(inputs)

    def forward(self, x, h_t, c_t):
        if h_t is None:
            h_t = self._init_state(x)
        if c_t is None:
            c_t = self._init_state(x)

        h_concat = self.conv_h_concat(h_t)
        i_h, g_h, f_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        ct_activation = torch.mul(c_t.repeat(1,2,1,1), self.ct_weight)
        i_c, f_c = torch.split(ct_activation, self.num_hidden, dim=1)

        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h

        if x != None:
            x_concat = self.conv_x_concat(x)
            i_x, g_x, f_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)

            i_ = i_ + i_x
            f_ = f_ + f_x
            g_ = g_ + g_x
            o_ = o_ + o_x

        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_ + self._forget_bias)
        c_new = f_ * c_t + i_ * torch.tanh(g_)

        o_c = torch.mul(c_new, self.oc_weight)

        h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)

        return h_new, c_new
