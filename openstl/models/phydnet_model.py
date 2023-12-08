import random
import torch
from torch import nn

from openstl.modules import PhyCell, PhyD_ConvLSTM, PhyD_EncoderRNN, K2M


class PhyDNet_Model(nn.Module):
    r"""PhyDNet Model

    Implementation of `Disentangling Physical Dynamics from Unknown Factors for
    Unsupervised Video Prediction <https://arxiv.org/abs/2003.01460>`_.

    """

    def __init__(self, configs, **kwargs):
        super(PhyDNet_Model, self).__init__()
        self.pre_seq_length = configs.pre_seq_length
        self.aft_seq_length = configs.aft_seq_length
        _, C, H, W = configs.in_shape
        patch_size = configs.patch_size if configs.patch_size in [2, 4] else 4
        input_shape = (H // patch_size, W // patch_size)

        self.phycell = PhyCell(input_shape=input_shape, input_dim=64, F_hidden_dims=[49],
                               n_layers=1, kernel_size=(7,7), device=configs.device)
        self.convcell = PhyD_ConvLSTM(input_shape=input_shape, input_dim=64, hidden_dims=[128,128,64],
                                      n_layers=3, kernel_size=(3,3), device=configs.device)
        self.encoder = PhyD_EncoderRNN(self.phycell, self.convcell,
                                       in_channel=C, patch_size=patch_size)
        self.k2m = K2M([7,7])

        self.criterion = nn.MSELoss()

    def forward(self, input_tensor, target_tensor, constraints, teacher_forcing_ratio=0.0):
        loss = 0
        for ei in range(self.pre_seq_length - 1):
            _, _, output_image, _, _ = self.encoder(input_tensor[:,ei,:,:,:], (ei==0))
            loss += self.criterion(output_image, input_tensor[:,ei+1,:,:,:])

        decoder_input = input_tensor[:,-1,:,:,:]
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        for di in range(self.aft_seq_length):
            _, _, output_image, _, _ = self.encoder(decoder_input)
            target = target_tensor[:,di,:,:,:]
            loss += self.criterion(output_image, target)
            if use_teacher_forcing:
                decoder_input = target
            else:
                decoder_input = output_image

        for b in range(0, self.encoder.phycell.cell_list[0].input_dim):
            filters = self.encoder.phycell.cell_list[0].F.conv1.weight[:,b,:,:]
            m = self.k2m(filters.double()).float()
            loss += self.criterion(m, constraints.to(m.device))

        return loss

    def inference(self, input_tensor, target_tensor, constraints, **kwargs):
        with torch.no_grad():
            loss = 0
            for ei in range(self.pre_seq_length - 1):
                encoder_output, encoder_hidden, output_image, _, _  = \
                    self.encoder(input_tensor[:,ei,:,:,:], (ei==0))
                if kwargs.get('return_loss', True):
                    loss += self.criterion(output_image, input_tensor[:,ei+1,:,:,:])

            decoder_input = input_tensor[:,-1,:,:,:]
            predictions = []

            for di in range(self.aft_seq_length):
                _, _, output_image, _, _ = self.encoder(decoder_input, False, False)
                decoder_input = output_image
                predictions.append(output_image)
                if kwargs.get('return_loss', True):
                    loss += self.criterion(output_image, target_tensor[:,di,:,:,:])

            for b in range(0, self.encoder.phycell.cell_list[0].input_dim):
                filters = self.encoder.phycell.cell_list[0].F.conv1.weight[:,b,:,:]
                m = self.k2m(filters.double()).float()
                if kwargs.get('return_loss', True):
                    loss += self.criterion(m, constraints.to(m.device))

            return torch.stack(predictions, dim=1), loss
