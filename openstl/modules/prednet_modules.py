import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.functional import hardsigmoid


class PredNetConvLSTMCell(nn.Module):
    """ConvLSTMCell: Implementation of a single Convolutional LSTM cell

        Arguments
        ---------

        input_size: (int, int)
            Image size (height, width)
        input_dim: int
            dim of input tensor
        hidden_dim: int
            dim of ouptut/hidde/cell tensor (number of kernels/channels)
        kernel_size: (int, int)
            dims of kernel
        gating_mode: str
            ['mul', 'sub']
        peephole: boolean
            To include/exclude peephole connections
        tied_bias: optional, boolean
            toggle between tied vs untied bias weights for Conv2d

        Example
        -------
        >> model =  ConvLSTMCell(3, 2, (3,3), bias =False)
    """

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, gating_mode='mul', 
                peephole=False, tied_bias=False):

        super(PredNetConvLSTMCell, self).__init__()
        """
        Input gate params:, Wxi, Whi, Wci

        Forget gate params: Wxf, Whf, Wcf

        Output gate params: Wxo, Who, Wco

        Candidate: Wxc, Whc
        """

        # Convolution hyper-parameters
        self.gating_mode = gating_mode
        self.peephole = peephole
        self.tied_bias = tied_bias

        self.input_size = input_size # used for peephole connections
        self.input_dim = input_dim # color channels in input (# of kernels in prev layer)
        self.hidden_dim = hidden_dim # of kernels in current layer
        self.kernel_size = kernel_size
        # Compatible with python 2.7
        self.padding = tuple(k // 2 for k in self.kernel_size)
        # Below works for python 3.6 
        # self.padding = 'same' # height x width of hidden state is determined by padding
        self.kern_names = ["Wxi", "Whi", "Wxf", "Whf", "Wxo", "Who", "Wxc", "Whc"]
        self.peep_names = ["Wci", "Wcf", "Wco"]
        self.tied_bias_names = ["bi", "bf", "bc", "bo"]

        for kern_n in self.kern_names:
            if 'x' in kern_n: # kernels that convolve input Xt
                self.__setattr__(kern_n, nn.Conv2d(self.input_dim, self.hidden_dim, self.kernel_size, 1,
                                self.padding, bias= not self.tied_bias))
            else: # kernels that convovel Ht or Ct
                self.__setattr__(kern_n, nn.Conv2d(self.hidden_dim, self.hidden_dim, self.kernel_size, 1,
                                self.padding, bias= not self.tied_bias))

        for peep_n in self.peep_names:
            self.register_parameter(peep_n, Parameter(
                torch.ones(self.hidden_dim, *self.input_size, requires_grad=self.peephole)))


        for bias in self.tied_bias_names:
            # a scalar (tied) bias for each kern
            self.register_parameter(bias, Parameter(
                torch.zeros((hidden_dim,1,1), requires_grad=self.tied_bias)))

    def forward(self, input_tensor, prev_state):
        """ forward a time-point through ConvLSTM cell

        Args:
            input_tensor: 4D Tensor Xt Dims: (N, C, H, W)
            prev_state: (4D Tensor, 4D Tensor) (Ht, Ct)
        """

        Xt = input_tensor
        Htm1, Ctm1 = prev_state

        i = hardsigmoid(self.Wxi(Xt) + self.Whi(Htm1)  + self.Wci * Ctm1 + self.bi)
        f = hardsigmoid(self.Wxf(Xt) + self.Whf(Htm1) + self.Wcf * Ctm1 + self.bf)
        if self.gating_mode == 'mul':
            Ct_ = torch.tanh(self.Wxc(Xt) + self.Whc(Htm1) + self.bc) # candidate for new cell state
            Ct = f * Ctm1 + i * Ct_
            o = hardsigmoid(self.Wxo(Xt) + self.Who(Htm1) + self.Wco * Ct + self.bo)
            Ht = o * torch.tanh(Ct)
        else:
            Ct_ = hardsigmoid(self.Wxc(Xt) + self.Whc(Htm1) + self.bc) # candidate for new cell state
            Ct = f * Ctm1 + Ct_ - i
            o = hardsigmoid(self.Wxo(Xt) + self.Who(Htm1) + self.Wco * Ct + self.bo)
            Ht = hardsigmoid(Ct) - o

        return (Ht, Ct)
