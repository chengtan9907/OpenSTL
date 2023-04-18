import torch
from torch import nn
from torch.autograd import Variable

from openstl.modules import zig_rev_predictor, autoencoder


class CrevNet_Model(nn.Module):
    r"""CrevNet Model

    Implementation of `Efficient and Information-Preserving Future Frame Prediction
    and Beyond <https://openreview.net/forum?id=B1eY_pVYvB>`_.
    """

    def __init__(self, in_shape, rnn_size, batch_size, predictor_rnn_layers,
                 pre_seq_length, aft_seq_length, n_eval, **kwargs):
        super(CrevNet_Model, self).__init__()
        T, channels, image_height, image_width = in_shape
        self.rnn_size = rnn_size
        self.n_eval = n_eval
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length

        self.frame_predictor = zig_rev_predictor(
            rnn_size, rnn_size, rnn_size, predictor_rnn_layers, batch_size)

        self.encoder = autoencoder(nBlocks=[4,5,3], nStrides=[1, 2, 2],
                            nChannels=None, init_ds=2,
                            dropout_rate=0., affineBN=True,
                            in_shape=[channels, image_height, image_width],
                            mult=2)
        self.criterion = nn.MSELoss()

    def forward(self, x, training=True):
        B, T, C, H, W = x.shape

        input = []
        for j in range(self.n_eval):
            k1 = x[:, j].unsqueeze(2)
            k2 = x[:, j + 1].unsqueeze(2)
            k3 = x[:, j + 2].unsqueeze(2)
            input.append(torch.cat((k1,k2,k3), 2))

        loss = 0
        self.frame_predictor.hidden = self.frame_predictor.init_hidden()
        memo = Variable(torch.zeros(B, self.rnn_size, 3, H // 8, W // 8).cuda())
        for i in range(1, self.pre_seq_length + self.aft_seq_length):
            h = self.encoder(input[i - 1], True)
            h_pred, memo = self.frame_predictor((h, memo))
            x_pred = self.encoder(h_pred, False)
            loss += (self.criterion(x_pred, input[i]))

        if training is True:
            return loss
        else:
            gen_seq = []
            self.frame_predictor.hidden = self.frame_predictor.init_hidden()
            memo = torch.zeros(B, self.rnn_size, 3, H // 8, W // 8).cuda()
            x_in = input[self.pre_seq_length-1]
            for i in range(self.pre_seq_length, self.n_eval):
                h = self.encoder(x_in)
                h_pred, memo = self.frame_predictor((h, memo))
                if i == self.pre_seq_length:
                    x_in = self.encoder(h_pred, False).detach()
                    x_in[:, :, 0] = input[i][:, :, 0]
                    x_in[:, :, 1] = input[i][:, :, 1]
                elif i == self.pre_seq_length + 1:
                    x_in = self.encoder(h_pred, False).detach()
                    x_in[:, :, 0] = input[i][:, :, 0]
                else:
                    x_in = self.encoder(h_pred, False).detach()
                gen_seq.append(x_in[:, 0, 2][:, None, ...])

            return torch.stack(gen_seq, dim=1), loss
