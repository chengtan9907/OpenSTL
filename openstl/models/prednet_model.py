import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from openstl.utils import get_initial_states


class PredNet_Model(nn.Module):
    def __init__(self, stack_sizes, R_stack_sizes,
                 A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                 pixel_max=1., args=None):
        super(PredNet_Model, self).__init__()
        self.args = args
        self.stack_sizes = stack_sizes
        self.num_layers = len(stack_sizes)
        assert len(R_stack_sizes) == self.num_layers
        self.R_stack_sizes = R_stack_sizes
        assert len(A_filt_sizes) == self.num_layers - 1
        self.A_filt_sizes = A_filt_sizes
        assert len(Ahat_filt_sizes) == self.num_layers
        self.Ahat_filt_sizes = Ahat_filt_sizes
        assert len(R_filt_sizes) == self.num_layers
        self.R_filt_sizes = R_filt_sizes

        self.pixel_max = pixel_max
        self.error_activation = args.error_activation  # 'relu'
        self.A_activation = args.A_activation  # 'relu'
        self.LSTM_activation = args.LSTM_activation  # 'tanh'
        self.LSTM_inner_activation = args.LSTM_inner_activation  # 'hard_sigmoid'
        self.channel_axis = -3
        self.row_axis = -2
        self.col_axis = -1

        self.get_activationFunc = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
        }

        self.build_layers()
        self.init_weights()

    def init_weights(self):
        def init_layer_weights(layer):
            if isinstance(layer, nn.Conv2d):
                layer.bias.data.zero_()
        self.apply(init_layer_weights)

    def hard_sigmoid(self, x, slope=0.2, shift=0.5):
        x = (slope * x) + shift
        x = torch.clamp(x, 0, 1)
        return x

    def batch_flatten(self, x):
        shape = [*x.size()]
        dim = np.prod(shape[1:])
        return x.view(-1, int(dim))

    def isNotTopestLayer(self, layerIndex):
        '''judge if the layerIndex is not the topest layer.'''
        return True if layerIndex < self.num_layers - 1 else False

    def build_layers(self):
        # i: input, f: forget, c: cell, o: output
        self.conv_layers = {item: []
                            for item in ['i', 'f', 'c', 'o', 'A', 'Ahat']}
        lstm_list = ['i', 'f', 'c', 'o']

        for item in sorted(self.conv_layers.keys()):
            for l in range(self.num_layers):
                if item == 'Ahat':
                    in_channels = self.R_stack_sizes[l]
                    self.conv_layers['Ahat'].append(nn.Conv2d(in_channels=in_channels,
                                                              out_channels=self.stack_sizes[
                                                                  l], kernel_size=self.Ahat_filt_sizes[l],
                                                              stride=(1, 1), padding=int((self.Ahat_filt_sizes[l] - 1) / 2)))
                    act = 'relu' if l == 0 else self.A_activation
                    self.conv_layers['Ahat'].append(self.get_activationFunc[act])
                elif item == 'A':
                    if self.isNotTopestLayer(l):
                        in_channels = self.R_stack_sizes[l] * 2
                        self.conv_layers['A'].append(nn.Conv2d(in_channels=in_channels,
                                                               out_channels=self.stack_sizes[l + 1], kernel_size=self.A_filt_sizes[l], stride=(1, 1), padding=int((self.A_filt_sizes[l] - 1) / 2)))
                        self.conv_layers['A'].append(self.get_activationFunc[self.A_activation])
                elif item in lstm_list:     # build R module
                    in_channels = self.stack_sizes[l] * \
                        2 + self.R_stack_sizes[l]
                    if self.isNotTopestLayer(l):
                        in_channels += self.R_stack_sizes[l + 1]
                    self.conv_layers[item].append(nn.Conv2d(in_channels=in_channels, out_channels=self.R_stack_sizes[l],
                                                  kernel_size=self.R_filt_sizes[l], stride=(1, 1), padding=int((self.R_filt_sizes[l] - 1) / 2)))

        for name, layerList in self.conv_layers.items():
            self.conv_layers[name] = nn.ModuleList(
                layerList).to(self.args.device)
            setattr(self, name, self.conv_layers[name])

        self.upSample = nn.Upsample(scale_factor=2, mode='nearest')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def step(self, A, states, extrapolation=False):
        n = self.num_layers
        R_current = states[:(n)]
        C_current = states[(n):(2 * n)]
        E_current = states[(2 * n):(3 * n)]

        timestep = states[-1]
        if extrapolation == True and timestep >= self.args.in_shape[0]:
            A = states[-2]

        R_list, C_list, E_list = [], [], []

        for l in reversed(range(self.num_layers)):
            inputs = [R_current[l], E_current[l]]
            if self.isNotTopestLayer(l):
                inputs.append(R_up)
            inputs = torch.cat(inputs, dim=self.channel_axis)

            in_gate = self.hard_sigmoid(self.conv_layers['i'][l](inputs))
            forget_gate = self.hard_sigmoid(self.conv_layers['f'][l](inputs))
            cell_gate = F.tanh(self.conv_layers['c'][l](inputs))
            out_gate = self.hard_sigmoid(self.conv_layers['o'][l](inputs))
            C_next = (forget_gate * C_current[l]) + (in_gate * cell_gate)
            R_next = out_gate * F.tanh(C_next)

            C_list.insert(0, C_next)
            R_list.insert(0, R_next)

            if l > 0:
                R_up = self.upSample(R_next)

        for l in range(self.num_layers):
            Ahat = self.conv_layers['Ahat'][2 * l](R_list[l])  # ConvLayer
            Ahat = self.conv_layers['Ahat'][2 *
                                            l + 1](Ahat)   # activation function

            if l == 0:
                Ahat = torch.clamp(Ahat, max=self.pixel_max)
                frame_prediction = Ahat

            if self.error_activation.lower() == 'relu':
                E_up = F.relu(Ahat - A)
                E_down = F.relu(A - Ahat)
            elif self.error_activation.lower() == 'tanh':
                E_up = F.tanh(Ahat - A)
                E_down = F.tanh(A - Ahat)
            else:
                raise (RuntimeError(
                    'cannot obtain the activation function named %s' % self.error_activation))

            E_list.append(torch.cat((E_up, E_down), dim=self.channel_axis))

            if self.isNotTopestLayer(l):
                A = self.conv_layers['A'][2 * l](E_list[l])
                A = self.conv_layers['A'][2 * l + 1](A)
                A = self.pool(A)    # target for next layer

        for l in range(self.num_layers):
            layer_error = torch.mean(self.batch_flatten(
                E_list[l]), dim=-1, keepdim=True)
            all_error = layer_error if l == 0 else torch.cat(
                (all_error, layer_error), dim=-1)

        states = R_list + C_list + E_list
        predict = frame_prediction
        error = all_error
        states += [frame_prediction, timestep + 1]
        return predict, error, states

    def forward(self, A0_withTimeStep, initial_states=None, extrapolation=False):
        '''
        A0_withTimeStep is the input from dataloader.
        Its shape is: (batch_size, timesteps, Channel, Height, Width).

        '''
        if initial_states is None:
            T, C, H, W = self.args.in_shape
            initial_states = get_initial_states((1, T, C, H, W)),
                                                self.row_axis, self.col_axis, self.num_layers, self.R_stack_sizes, self.stack_sizes, self.channel_axis, self.args.device)
        A0_withTimeStep = A0_withTimeStep.transpose(0, 1)
        num_timesteps = A0_withTimeStep.shape[0]

        hidden_states = initial_states
        predict_list, error_list = [], []
        for t in range(num_timesteps):
            A0 = A0_withTimeStep[t, ...]
            predict, error, hidden_states = self.step(
                A0, hidden_states, extrapolation)
            predict_list.append(predict)
            error_list.append(error)
        return predict_list, error_list
