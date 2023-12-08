import torch
import torch.nn as nn


class CausalLSTMCell(nn.Module):

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(CausalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 7, height, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_c = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, height, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, height, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, height, width])
            )
            self.conv_c2m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_om = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, height, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_c = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_c2m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_om = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1,
                                   stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        c_concat = self.conv_c(c_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = \
            torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, m_m = torch.split(m_concat, self.num_hidden, dim=1)
        i_c, f_c, g_c = torch.split(c_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h + i_c)
        f_t = torch.sigmoid(f_x + f_h + f_c + self._forget_bias)
        g_t = torch.tanh(g_x + g_h + g_c)

        c_new = f_t * c_t + i_t * g_t

        c2m = self.conv_c2m(c_new)
        i_c, g_c, f_c, o_c = torch.split(c2m, self.num_hidden, dim=1)

        i_t_prime = torch.sigmoid(i_x_prime + i_m + i_c)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + f_c + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_c)

        m_new = f_t_prime * torch.tanh(m_m) + i_t_prime * g_t_prime
        o_m = self.conv_om(m_new)

        o_t = torch.tanh(o_x + o_h + o_c + o_m)
        mem = torch.cat((c_new, m_new), 1)
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new


class GHU(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size,
                 stride, layer_norm, initializer=0.001):
        super(GHU, self).__init__()

        self.filter_size = filter_size
        self.padding = filter_size // 2
        self.num_hidden = num_hidden
        self.layer_norm = layer_norm

        if layer_norm:
            self.z_concat = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 2, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, height, width])
            )
            self.x_concat = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 2, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, height, width])
            )
        else:
            self.z_concat = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 2, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.x_concat = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 2, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )


        if initializer != -1:
            self.initializer = initializer
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            nn.init.uniform_(m.weight, -self.initializer, self.initializer)

    def _init_state(self, inputs):
        return torch.zeros_like(inputs)

    def forward(self, x, z):
        if z is None:
            z = self._init_state(x)
        z_concat = self.z_concat(z)
        x_concat = self.x_concat(x)

        gates = x_concat + z_concat
        p, u = torch.split(gates, self.num_hidden, dim=1)
        p = torch.tanh(p)
        u = torch.sigmoid(u)
        z_new = u * p + (1-u) * z
        return z_new
