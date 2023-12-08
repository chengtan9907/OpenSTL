import torch
import torch.nn as nn
from numpy import *
from numpy.linalg import *
from scipy.special import factorial
from functools import reduce

__all__ = ['M2K','K2M']


class PhyCell_Cell(nn.Module):

    def __init__(self, input_dim, F_hidden_dim, kernel_size, bias=1):
        super(PhyCell_Cell, self).__init__()
        self.input_dim  = input_dim
        self.F_hidden_dim = F_hidden_dim
        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.F = nn.Sequential()
        self.F.add_module('conv1', nn.Conv2d(in_channels=input_dim, out_channels=F_hidden_dim,
                                             kernel_size=self.kernel_size, stride=(1,1), padding=self.padding))
        self.F.add_module('bn1',nn.GroupNorm(7 ,F_hidden_dim))        
        self.F.add_module('conv2', nn.Conv2d(in_channels=F_hidden_dim, out_channels=input_dim,
                                             kernel_size=(1,1), stride=(1,1), padding=(0,0)))

        self.convgate = nn.Conv2d(in_channels=self.input_dim + self.input_dim,
                                  out_channels=self.input_dim,
                                  kernel_size=(3,3),
                                  padding=(1,1), bias=self.bias)

    def forward(self, x, hidden):  # x [batch_size, hidden_dim, height, width]
        combined = torch.cat([x, hidden], dim=1)  # concatenate along channel axis
        combined_conv = self.convgate(combined)
        K = torch.sigmoid(combined_conv)
        hidden_tilde = hidden + self.F(hidden)  # prediction
        next_hidden = hidden_tilde + K * (x-hidden_tilde)  # correction , Haddamard product
        return next_hidden


class PhyCell(nn.Module):

    def __init__(self, input_shape, input_dim, F_hidden_dims, n_layers, kernel_size, device):
        super(PhyCell, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.F_hidden_dims = F_hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H = []  
        self.device = device
             
        cell_list = []
        for i in range(0, self.n_layers):
            cell_list.append(PhyCell_Cell(input_dim=input_dim,
                                          F_hidden_dim=self.F_hidden_dims[i],
                                          kernel_size=self.kernel_size))
        self.cell_list = nn.ModuleList(cell_list)
        
       
    def forward(self, input_, first_timestep=False):  # input_ [batch_size, 1, channels, width, height]
        batch_size = input_.data.size()[0]
        if (first_timestep):   
            self.initHidden(batch_size)  # init Hidden at each forward start
        for j, cell in enumerate(self.cell_list):
            self.H[j] = self.H[j].to(input_.device)
            if j==0:  # bottom layer
                self.H[j] = cell(input_, self.H[j])
            else:
                self.H[j] = cell(self.H[j-1],self.H[j])
        return self.H, self.H
    
    def initHidden(self, batch_size):
        self.H = [] 
        for i in range(self.n_layers):
            self.H.append(torch.zeros(
                batch_size, self.input_dim, self.input_shape[0], self.input_shape[1]).to(self.device))

    def setHidden(self, H):
        self.H = H


class PhyD_ConvLSTM_Cell(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size, bias=1):
        """
        input_shape: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(PhyD_ConvLSTM_Cell, self).__init__()
        
        self.height, self.width = input_shape
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding, bias=self.bias)
                 
    # we implement LSTM that process only one timestep 
    def forward(self, x, hidden): # x [batch, hidden_dim, width, height]
        h_cur, c_cur = hidden
        
        combined = torch.cat([x, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class PhyD_ConvLSTM(nn.Module):

    def __init__(self, input_shape, input_dim, hidden_dims, n_layers, kernel_size, device):
        super(PhyD_ConvLSTM, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H, self.C = [], []   
        self.device = device
        
        cell_list = []
        for i in range(0, self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i-1]
            print('layer ', i, 'input dim ', cur_input_dim, ' hidden dim ', self.hidden_dims[i])
            cell_list.append(PhyD_ConvLSTM_Cell(input_shape=self.input_shape,
                                                input_dim=cur_input_dim,
                                                hidden_dim=self.hidden_dims[i],
                                                kernel_size=self.kernel_size))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_, first_timestep=False): # input_ [batch_size, 1, channels, width, height]
        batch_size = input_.data.size()[0]
        if (first_timestep):   
            self.initHidden(batch_size) # init Hidden at each forward start
        for j, cell in enumerate(self.cell_list):
            self.H[j] = self.H[j].to(input_.device)
            self.C[j] = self.C[j].to(input_.device)
            if j==0: # bottom layer
                self.H[j], self.C[j] = cell(input_, (self.H[j],self.C[j]))
            else:
                self.H[j], self.C[j] = cell(self.H[j-1],(self.H[j],self.C[j]))
        return (self.H,self.C) , self.H   # (hidden, output)
    
    def initHidden(self,batch_size):
        self.H, self.C = [],[]  
        for i in range(self.n_layers):
            self.H.append(torch.zeros(
                batch_size, self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).to(self.device))
            self.C.append(torch.zeros(
                batch_size, self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).to(self.device))
    
    def setHidden(self, hidden):
        H,C = hidden
        self.H, self.C = H,C
 

class dcgan_conv(nn.Module):

    def __init__(self, nin, nout, stride):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=(3,3),
                          stride=stride, padding=1),
                nn.GroupNorm(16, nout),
                nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, input):
        return self.main(input)


class dcgan_upconv(nn.Module):

    def __init__(self, nin, nout, stride):
        super(dcgan_upconv, self).__init__()
        if stride==2:
            output_padding = 1
        else:
            output_padding = 0
        self.main = nn.Sequential(
                nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=(3,3),
                                   stride=stride, padding=1, output_padding=output_padding),
                nn.GroupNorm(16, nout),
                nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, input):
        return self.main(input)


class encoder_E(nn.Module):

    def __init__(self, nc=1, nf=32, patch_size=4):
        super(encoder_E, self).__init__()
        assert patch_size in [2, 4]
        stride_2 = patch_size // 2
        # input is (1) x 64 x 64
        self.c1 = dcgan_conv(nc, nf, stride=2) # (32) x 32 x 32
        self.c2 = dcgan_conv(nf, nf, stride=1) # (32) x 32 x 32
        self.c3 = dcgan_conv(nf, 2*nf, stride=stride_2) # (64) x 16 x 16

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        return h3


class decoder_D(nn.Module):

    def __init__(self, nc=1, nf=32, patch_size=4):
        super(decoder_D, self).__init__()
        assert patch_size in [2, 4]
        stride_2 = patch_size // 2
        output_padding = 1 if stride_2==2 else 0
        self.upc1 = dcgan_upconv(2*nf, nf, stride=2) #(32) x 32 x 32
        self.upc2 = dcgan_upconv(nf, nf, stride=1) #(32) x 32 x 32
        self.upc3 = nn.ConvTranspose2d(in_channels=nf, out_channels=nc, kernel_size=(3,3),
                                       stride=stride_2, padding=1,
                                       output_padding=output_padding)  #(nc) x 64 x 64

    def forward(self, input):
        d1 = self.upc1(input)
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        return d3


class encoder_specific(nn.Module):

    def __init__(self, nc=64, nf=64):
        super(encoder_specific, self).__init__()
        self.c1 = dcgan_conv(nc, nf, stride=1) # (64) x 16 x 16
        self.c2 = dcgan_conv(nf, nf, stride=1) # (64) x 16 x 16

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        return h2


class decoder_specific(nn.Module):

    def __init__(self, nc=64, nf=64):
        super(decoder_specific, self).__init__()
        self.upc1 = dcgan_upconv(nf, nf, stride=1) #(64) x 16 x 16
        self.upc2 = dcgan_upconv(nf, nc, stride=1) #(32) x 32 x 32
        
    def forward(self, input):
        d1 = self.upc1(input)
        d2 = self.upc2(d1)
        return d2


class PhyD_EncoderRNN(torch.nn.Module):

    def __init__(self, phycell, convcell, in_channel=1, patch_size=4):
        super(PhyD_EncoderRNN, self).__init__()
        self.encoder_E = encoder_E(nc=in_channel, patch_size=patch_size) # general encoder 64x64x1 -> 32x32x32
        self.encoder_Ep = encoder_specific() # specific image encoder 32x32x32 -> 16x16x64
        self.encoder_Er = encoder_specific()
        self.decoder_Dp = decoder_specific() # specific image decoder 16x16x64 -> 32x32x32
        self.decoder_Dr = decoder_specific()
        self.decoder_D = decoder_D(nc=in_channel, patch_size=patch_size) # general decoder 32x32x32 -> 64x64x1

        self.phycell = phycell
        self.convcell = convcell

    def forward(self, input, first_timestep=False, decoding=False):
        input = self.encoder_E(input) # general encoder 64x64x1 -> 32x32x32
    
        if decoding:  # input=None in decoding phase
            input_phys = None
        else:
            input_phys = self.encoder_Ep(input)
        input_conv = self.encoder_Er(input)     

        hidden1, output1 = self.phycell(input_phys, first_timestep)
        hidden2, output2 = self.convcell(input_conv, first_timestep)

        decoded_Dp = self.decoder_Dp(output1[-1])
        decoded_Dr = self.decoder_Dr(output2[-1])
        
        out_phys = torch.sigmoid(self.decoder_D(decoded_Dp)) # partial reconstructions for vizualization
        out_conv = torch.sigmoid(self.decoder_D(decoded_Dr))

        concat = decoded_Dp + decoded_Dr   
        output_image = torch.sigmoid( self.decoder_D(concat ))
        return out_phys, hidden1, output_image, out_phys, out_conv


def _apply_axis_left_dot(x, mats):
    assert x.dim() == len(mats)+1
    sizex = x.size()
    k = x.dim()-1
    for i in range(k):
        x = tensordot(mats[k-i-1], x, dim=[1,k])
    x = x.permute([k,]+list(range(k))).contiguous()
    x = x.view(sizex)
    return x

def _apply_axis_right_dot(x, mats):
    assert x.dim() == len(mats)+1
    sizex = x.size()
    k = x.dim()-1
    x = x.permute(list(range(1,k+1))+[0,])
    for i in range(k):
        x = tensordot(x, mats[i], dim=[0,0])
    x = x.contiguous()
    x = x.view(sizex)
    return x

class _MK(nn.Module):
    def __init__(self, shape):
        super(_MK, self).__init__()
        self._size = torch.Size(shape)
        self._dim = len(shape)
        M = []
        invM = []
        assert len(shape) > 0
        j = 0
        for l in shape:
            M.append(zeros((l,l)))
            for i in range(l):
                M[-1][i] = ((arange(l)-(l-1)//2)**i)/factorial(i)
            invM.append(inv(M[-1]))
            self.register_buffer('_M'+str(j), torch.from_numpy(M[-1]))
            self.register_buffer('_invM'+str(j), torch.from_numpy(invM[-1]))
            j += 1

    @property
    def M(self):
        return list(self._buffers['_M'+str(j)] for j in range(self.dim()))
    @property
    def invM(self):
        return list(self._buffers['_invM'+str(j)] for j in range(self.dim()))

    def size(self):
        return self._size
    def dim(self):
        return self._dim
    def _packdim(self, x):
        assert x.dim() >= self.dim()
        if x.dim() == self.dim():
            x = x[newaxis,:]
        x = x.contiguous()
        x = x.view([-1,]+list(x.size()[-self.dim():]))
        return x

    def forward(self):
        pass


class M2K(_MK):
    """
    convert moment matrix to convolution kernel
    Arguments:
        shape (tuple of int): kernel shape
    Usage:
        m2k = M2K([5,5])
        m = torch.randn(5,5,dtype=torch.float64)
        k = m2k(m)
    """
    def __init__(self, shape):
        super(M2K, self).__init__(shape)
    def forward(self, m):
        """
        m (Tensor): torch.size=[...,*self.shape]
        """
        sizem = m.size()
        m = self._packdim(m)
        m = _apply_axis_left_dot(m, self.invM)
        m = m.view(sizem)
        return m


class K2M(_MK):
    """
    convert convolution kernel to moment matrix
    Arguments:
        shape (tuple of int): kernel shape
    Usage:
        k2m = K2M([5,5])
        k = torch.randn(5,5,dtype=torch.float64)
        m = k2m(k)
    """
    def __init__(self, shape):
        super(K2M, self).__init__(shape)
    def forward(self, k):
        """
        k (Tensor): torch.size=[...,*self.shape]
        """
        sizek = k.size()
        k = self._packdim(k)
        k = _apply_axis_left_dot(k, self.M)
        k = k.view(sizek)
        return k


def tensordot(a,b,dim):
    """
    tensordot in PyTorch, see numpy.tensordot?
    """
    l = lambda x,y:x*y
    if isinstance(dim,int):
        a = a.contiguous()
        b = b.contiguous()
        sizea = a.size()
        sizeb = b.size()
        sizea0 = sizea[:-dim]
        sizea1 = sizea[-dim:]
        sizeb0 = sizeb[:dim]
        sizeb1 = sizeb[dim:]
        N = reduce(l, sizea1, 1)
        assert reduce(l, sizeb0, 1) == N
    else:
        adims = dim[0]
        bdims = dim[1]
        adims = [adims,] if isinstance(adims, int) else adims
        bdims = [bdims,] if isinstance(bdims, int) else bdims
        adims_ = set(range(a.dim())).difference(set(adims))
        adims_ = list(adims_)
        adims_.sort()
        perma = adims_+adims
        bdims_ = set(range(b.dim())).difference(set(bdims))
        bdims_ = list(bdims_)
        bdims_.sort()
        permb = bdims+bdims_
        a = a.permute(*perma).contiguous()
        b = b.permute(*permb).contiguous()

        sizea = a.size()
        sizeb = b.size()
        sizea0 = sizea[:-len(adims)]
        sizea1 = sizea[-len(adims):]
        sizeb0 = sizeb[:len(bdims)]
        sizeb1 = sizeb[len(bdims):]
        N = reduce(l, sizea1, 1)
        assert reduce(l, sizeb0, 1) == N
    a = a.view([-1,N])
    b = b.view([N,-1])
    c = a@b
    return c.view(sizea0+sizeb1)
