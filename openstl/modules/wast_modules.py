import torch, pywt
import torch.nn as nn
from einops import rearrange
from functools import partial
from itertools import accumulate
from timm.models.layers import DropPath, activations
from timm.models.efficientnet_blocks import SqueezeExcite, InvertedResidual

# version adaptation for PyTorch > 1.7.1
IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft

class HighFocalFrequencyLoss(nn.Module):
    """ Example:
        fake = torch.randn(4, 3, 128, 64)
        real = torch.randn(4, 3, 128, 64)
        hffl = HighFocalFrequencyLoss()

        loss = hffl(fake, real)
        print(loss)
    """

    def __init__(self, loss_weight=0.001, level=1, tau=0.1, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=True, batch_matrix=False):
        super(HighFocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix
        self.level = level
        self.tau = tau
        self.DWT = WaveletTransform2D().cuda()

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(y, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(y, 2, onesided=False, normalized=True)
        return freq

    def build_freq_mask(self, shape):
        H, W = shape[-2:]
        radius = self.tau * max(H, W)
        Y, X = torch.meshgrid(torch.arange(H), torch.arange(W))

        mask = torch.ones_like(X, dtype=torch.float32).cuda()

        centers = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]

        for center in centers:
            distance = torch.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)
            mask[distance <= radius] = 0
        return mask

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        mask = self.build_freq_mask(weight_matrix.shape)
        loss = weight_matrix * freq_distance * mask
        return torch.mean(loss)

    def frequency_loss(self, pred, target, matrix=None):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        return self.loss_formulation(pred_freq, target_freq, matrix)

    def forward(self, pred, target, matrix=None, **kwargs):
        pred = rearrange(pred, 'b t c h w -> (b t) c h w') if kwargs["reshape"] is True else pred
        target = rearrange(target, 'b t c h w -> (b t) c h w') if kwargs["reshape"] is True else target

        loss = 0
        for level in range(self.level):
            pred, _, _, _ = self.DWT(pred)
            target, _, _, _ = self.DWT(target)
            loss += self.frequency_loss(pred, target, matrix)
        return loss * self.loss_weight


class WaveletTransform2D(nn.Module):
    """Compute a two-dimensional wavelet transform.
        loss = nn.MSELoss()
        data = torch.rand(1, 3, 128, 256)
        DWT = WaveletTransform2D()
        IDWT = WaveletTransform2D(inverse=True)

        LL, LH, HL, HH = DWT(data)
        recdata = IDWT([LL, LH, HL, HH])
        print(loss(data, recdata))
    """
    def __init__(self, inverse=False, wavelet="haar", mode="constant"):
        super(WaveletTransform2D, self).__init__()
        self.mode = mode
        wavelet = pywt.Wavelet(wavelet)

        if isinstance(wavelet, tuple):
            dec_lo, dec_hi, rec_lo, rec_hi = wavelet
        else:
            dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank

        self.inverse = inverse
        if inverse is False:
            dec_lo = torch.tensor(dec_lo).flip(-1).unsqueeze(0)
            dec_hi = torch.tensor(dec_hi).flip(-1).unsqueeze(0)
            self.build_filters(dec_lo, dec_hi)
        else:
            rec_lo = torch.tensor(rec_lo).unsqueeze(0)
            rec_hi = torch.tensor(rec_hi).unsqueeze(0)
            self.build_filters(rec_lo, rec_hi)

    def build_filters(self, lo, hi):
        # construct 2d filter
        self.dim_size = lo.shape[-1]
        ll = self.outer(lo, lo)
        lh = self.outer(hi, lo)
        hl = self.outer(lo, hi)
        hh = self.outer(hi, hi)
        filters = torch.stack([ll, lh, hl, hh],dim=0)
        filters = filters.unsqueeze(1)
        self.register_buffer('filters', filters)  # [4, 1, height, width]

    def outer(self, a: torch.Tensor, b: torch.Tensor):
        """Torch implementation of numpy's outer for 1d vectors."""
        a_flat = torch.reshape(a, [-1])
        b_flat = torch.reshape(b, [-1])
        a_mul = torch.unsqueeze(a_flat, dim=-1)
        b_mul = torch.unsqueeze(b_flat, dim=0)
        return a_mul * b_mul

    def get_pad(self, data_len: int, filter_len: int):
        padr = (2 * filter_len - 3) // 2
        padl = (2 * filter_len - 3) // 2
        # pad to even singal length.
        if data_len % 2 != 0:
            padr += 1
        return padr, padl

    def adaptive_pad(self, data):
        padb, padt = self.get_pad(data.shape[-2], self.dim_size)
        padr, padl = self.get_pad(data.shape[-1], self.dim_size)

        data_pad = torch.nn.functional.pad(data, [padl, padr, padt, padb], mode=self.mode)
        return data_pad

    def forward(self, data):
        if self.inverse is False:
            b, c, h, w = data.shape
            dec_res = []
            data = self.adaptive_pad(data)
            for filter in self.filters:
                dec_res.append(torch.nn.functional.conv2d(data, filter.repeat(c, 1, 1, 1), stride=2, groups=c))
            return dec_res
        else:
            b, c, h, w = data[0].shape
            data = torch.stack(data, dim=2).reshape(b, -1, h, w)
            rec_res = torch.nn.functional.conv_transpose2d(data, self.filters.repeat(c, 1, 1, 1), stride=2, groups=c)
            return rec_res


class WaveletTransform3D(nn.Module):
    """Compute a three-dimensional wavelet transform.
        Example:
            loss = nn.MSELoss()
            data = torch.rand(1, 3, 10, 128, 256)
            DWT = WaveletTransform3D()
            IDWT = WaveletTransform3D(inverse=True)

            LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = DWT(data)
            recdata = IDWT([LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH])
            print(loss(data, recdata))

            LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = DWT_3D(data)
            recdata = IDWT_3D(LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH)
            print(loss(data, recdata))
        """
    def __init__(self, inverse=False, wavelet="haar", mode="constant"):
        super(WaveletTransform3D, self).__init__()
        self.mode = mode
        wavelet = pywt.Wavelet(wavelet)

        if isinstance(wavelet, tuple):
            dec_lo, dec_hi, rec_lo, rec_hi = wavelet
        else:
            dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank

        self.inverse = inverse
        if inverse is False:
            dec_lo = torch.tensor(dec_lo).flip(-1).unsqueeze(0)
            dec_hi = torch.tensor(dec_hi).flip(-1).unsqueeze(0)
            self.build_filters(dec_lo, dec_hi)
        else:
            rec_lo = torch.tensor(rec_lo).unsqueeze(0)
            rec_hi = torch.tensor(rec_hi).unsqueeze(0)
            self.build_filters(rec_lo, rec_hi)

    def build_filters(self, lo, hi):
        # construct 3d filter
        self.dim_size = lo.shape[-1]
        size = [self.dim_size] * 3
        lll = self.outer(lo, self.outer(lo, lo)).reshape(size)
        llh = self.outer(lo, self.outer(lo, hi)).reshape(size)
        lhl = self.outer(lo, self.outer(hi, lo)).reshape(size)
        lhh = self.outer(lo, self.outer(hi, hi)).reshape(size)
        hll = self.outer(hi, self.outer(lo, lo)).reshape(size)
        hlh = self.outer(hi, self.outer(lo, hi)).reshape(size)
        hhl = self.outer(hi, self.outer(hi, lo)).reshape(size)
        hhh = self.outer(hi, self.outer(hi, hi)).reshape(size)
        filters = torch.stack([lll, llh, lhl, lhh, hll, hlh, hhl, hhh], dim=0)
        filters = filters.unsqueeze(1)
        self.register_buffer('filters', filters)  # [8, 1, length, height, width]
        
    def outer(self, a: torch.Tensor, b: torch.Tensor):
        """Torch implementation of numpy's outer for 1d vectors."""
        a_flat = torch.reshape(a, [-1])
        b_flat = torch.reshape(b, [-1])
        a_mul = torch.unsqueeze(a_flat, dim=-1)
        b_mul = torch.unsqueeze(b_flat, dim=0)
        return a_mul * b_mul

    def get_pad(self, data_len: int, filter_len: int):
        padr = (2 * filter_len - 3) // 2
        padl = (2 * filter_len - 3) // 2
        # pad to even singal length.
        if data_len % 2 != 0:
            padr += 1
        return padr, padl

    def adaptive_pad(self, data):
        pad_back, pad_front = self.get_pad(data.shape[-3], self.dim_size)
        pad_bottom, pad_top = self.get_pad(data.shape[-2], self.dim_size)
        pad_right, pad_left = self.get_pad(data.shape[-1], self.dim_size)
        data_pad = torch.nn.functional.pad(
            data, [pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back], mode=self.mode)
        return data_pad

    def forward(self, data):
        if self.inverse is False:
            b, c, t, h, w = data.shape
            dec_res = []
            data = self.adaptive_pad(data)
            for filter in self.filters:
                dec_res.append(torch.nn.functional.conv3d(data, filter.repeat(c, 1, 1, 1, 1), stride=2, groups=c))
            return dec_res
        else:
            b, c, t, h, w = data[0].shape
            data = torch.stack(data, dim=2).reshape(b, -1, t, h, w)
            rec_res = torch.nn.functional.conv_transpose3d(data, self.filters.repeat(c, 1, 1, 1, 1), stride=2, groups=c)
            return rec_res


class FrequencyAttention(nn.Module):
    def __init__(self, in_dim, out_dim, reduction=32):
        super(FrequencyAttention, self).__init__()
        self.avgpool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avgpool_w = nn.AdaptiveAvgPool2d((1, None))

        hidden_dim = max(8, in_dim // reduction)

        self.conv1 = nn.Conv2d(in_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act = activations.HardSwish(inplace=True)

        self.conv_h = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.avgpool_h(x)  # b c h 1
        x_w = self.avgpool_w(x).permute(0, 1, 3, 2)  # b c w 1

        y = torch.cat([x_h, x_w], dim=2)  # b c (h+w) 1
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class TF_AwareBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., ls_init_value=1e-2, drop_path=0.1, large_kernel=51, small_kernel=5):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
 
        self.lk1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(large_kernel, 5), groups=dim, padding="same"),
            nn.BatchNorm2d(dim)
        )

        self.lk2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(5, large_kernel), groups=dim, padding="same"),
            nn.BatchNorm2d(dim)
        )

        self.sk = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(small_kernel, small_kernel), groups=dim, padding="same"),
            nn.BatchNorm2d(dim)
        )

        self.low_frequency_attn = FrequencyAttention(in_dim=dim, out_dim=dim, reduction=4)
        self.high_frequency_attn = FrequencyAttention(in_dim=dim, out_dim=dim, reduction=4)

        self.temporal_mixer = InvertedResidual(in_chs=dim, out_chs=dim, dw_kernel_size=7, exp_ratio=mlp_ratio,
                                            se_layer=partial(SqueezeExcite, rd_ratio=0.25), noskip=True)


        self.layer_scale_1 = nn.Parameter(ls_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(ls_init_value * torch.ones((dim)), requires_grad=True)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2'}

    def forward(self, x):
        attn = self.norm1(x)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * (self.low_frequency_attn(self.lk1(attn) + self.lk2(attn)) + self.high_frequency_attn(self.sk(attn))))
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.temporal_mixer(self.norm2(x)))
        return x


class TF_AwareBlocks(nn.Module):
    def __init__(self, dim, num_blocks, drop_path, use_bottleneck=None, use_hid=False, mlp_ratio=4., drop=0., ls_init_value=1e-2, large_kernel=51, small_kernel=5):
        super().__init__()
        assert len(drop_path) == num_blocks, "drop_path list doesn't match num_blocks"
        self.use_hid = use_hid
        self.use_bottleneck = use_bottleneck

        blocks = []
        for i in range(num_blocks):
            block = TF_AwareBlock(dim, mlp_ratio, drop, ls_init_value, drop_path[i], large_kernel, small_kernel)
            blocks.append(block)
        self.blocks = nn.Sequential(*blocks)
        self.concat_block = nn.Conv2d(dim * 2, dim, 3, 1, 1) if use_hid==True else None

        self.DWT = WaveletTransform3D(inverse=False) if use_bottleneck == "decompose" else None
        self.IDWT = WaveletTransform3D(inverse=True) if use_bottleneck == "decompose" else None

    def forward(self, x, skip=None):  # b, c ,t, h, w
        if self.concat_block is not None and self.use_bottleneck is None:
            b, c, t, h, w = x.shape
            x = rearrange(x, 'b c t h w -> b (c t) h w')
            x = self.concat_block(torch.cat([x, skip], dim=1))
            x = self.blocks(x)
            x = rearrange(x, 'b (c t) h w -> b c t h w', t=t)
            return x
        elif self.concat_block is None and self.use_bottleneck is None:
            b, c, t, h, w = x.shape
            x = rearrange(x, 'b c t h w -> b (c t) h w')
            x = skip= self.blocks(x)
            x = rearrange(x, 'b (c t) h w -> b c t h w', t=t)
            return x, skip
        elif self.use_bottleneck is not None:
            LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.DWT(x) if self.use_bottleneck == "decompose" else [x, None, None, None, None, None, None, None]
            b, c, t, h, w = LLL.shape
            LLL = rearrange(LLL, 'b c t h w -> b (c t) h w')
            LLL = self.blocks(LLL)
            LLL = rearrange(LLL, 'b (c t) h w -> b c t h w', t=t)
            x = self.IDWT([LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH]) if self.use_bottleneck == "decompose" else LLL
            return x



class Wavelet_3D_Embedding(nn.Module):
    def __init__(self, in_dim, out_dim, emb_dim=None):
        super().__init__()
        emb_dim = in_dim if emb_dim==None else emb_dim
        self.conv_0 = nn.Sequential(nn.Conv3d(in_dim, in_dim, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),),
                    nn.BatchNorm3d(in_dim),
                    nn.GELU(),)
        self.conv_1 = nn.Sequential(nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),),
                    nn.BatchNorm3d(out_dim),
                    nn.GELU(),)

        self.conv_emb = nn.Conv3d(emb_dim * 4, out_dim, kernel_size=(3, 3, 3),stride=(1, 1, 1),padding=(1, 1, 1),)

        self.DWT = WaveletTransform3D(inverse=False)

    def forward(self, x, x_emb=None):
        # embedding branch
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.DWT(x_emb)
        lo_temp = torch.cat([LLL, LHL, HLL, HHL], dim=1)
        hi_temp = torch.cat([LLH, LHH, HLH, HHH], dim=1)
        x_emb = torch.cat([lo_temp, hi_temp], dim=2)
        x_emb = self.conv_emb(x_emb)
        # downsampling branch
        x = self.conv_0(x)
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.DWT(x)
        spatio_lo_coeffs = torch.cat([LLL, LLH], dim=2)
        spatio_hi_coeffs = torch.cat([LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
        x = self.conv_1(spatio_lo_coeffs)
        return (x + x_emb), spatio_hi_coeffs


class Wavelet_3D_Reconstruction(nn.Module):
    def __init__(self, in_dim, out_dim, hi_dim):
        super().__init__()
        self.conv_0 = nn.Sequential(nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),),
            nn.BatchNorm3d(out_dim),
            nn.GELU(),)

        self.conv_hi =  nn.Sequential(nn.Conv3d(int(hi_dim * 6), int(out_dim * 6), kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=6),
            nn.BatchNorm3d(out_dim * 6),
            nn.GELU(),)

        self.IDWT = WaveletTransform3D(inverse=True)

    def forward(self, x, skip_hi=None):
        LLL, LLH = torch.chunk(self.conv_0(x), chunks=2, dim=2)
        LHL, LHH, HLL, HLH, HHL, HHH = torch.chunk(self.conv_hi(skip_hi), chunks=6, dim=1)
        x = self.IDWT([LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH])
        return x


class WaST_level1(nn.Module):
    def __init__(self, in_shape, encoder_dim, block_list=[2, 2, 2], drop_path_rate=0.1, mlp_ratio=4., **kwargs):
        super().__init__()
        frame, in_dim, H, W = in_shape
        self.block_list = block_list
        dp_list = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.block_list))]
        indexes = list(accumulate(block_list))
        dp_list = [dp_list[start:end] for start, end in zip([0] + indexes, indexes)]

        self.conv_in = nn.Sequential(
                    nn.Conv3d(
                        in_dim,
                        encoder_dim,
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                    ),
                    nn.BatchNorm3d(encoder_dim),
                    nn.GELU()
                )
        self.translator1 = TF_AwareBlocks(dim=encoder_dim * frame, num_blocks=block_list[0], drop_path=dp_list[0], mlp_ratio=mlp_ratio, large_kernel=51, small_kernel=5)

        self.wavelet_embed1 = Wavelet_3D_Embedding(in_dim=encoder_dim, out_dim=encoder_dim * 2, emb_dim=in_dim)  # wavelet_recon2: hi_dim = in_dim

        self.bottleneck_translator = TF_AwareBlocks(dim=encoder_dim * 2 * frame, num_blocks=block_list[1], drop_path=dp_list[1], use_bottleneck=True, mlp_ratio=mlp_ratio, large_kernel=21, small_kernel=5)

        self.wavelet_recon1 = Wavelet_3D_Reconstruction(in_dim=encoder_dim * 2, out_dim=encoder_dim, hi_dim=encoder_dim)
        self.translator2 = TF_AwareBlocks(dim=encoder_dim * frame, num_blocks=block_list[2], drop_path=dp_list[2], use_hid=True, mlp_ratio=mlp_ratio, large_kernel=51, small_kernel=5)

        self.conv_out = nn.Sequential(
                    nn.BatchNorm3d(encoder_dim),
                    nn.GELU(),
                    nn.Conv3d(
                        encoder_dim,
                        in_dim,
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1))
        )

    def update_drop_path(self, drop_path_rate):
        dp_list = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.block_list))]
        indexes = list(accumulate(self.block_list))
        dp_lists = [dp_list[start:end] for start, end in zip([0] + indexes, indexes)]
        dp_apply_blocks = [self.translator1.blocks, self.bottleneck_translator.blocks, self.translator2.blocks]
        for translators, dp_list_translators in zip(dp_apply_blocks, dp_lists):
            for translator, dp_list_translator in zip(translators, dp_list_translators):
                translator.drop_path.drop_prob = dp_list_translator

    def forward(self, x):
        x = rearrange(x, 'b t c h w -> b c t h w')

        ori_img = x
        x = self.conv_in(x)

        x, tskip1 = self.translator1(x)
        x, skip1 = self.wavelet_embed1(x, x_emb=ori_img)

        x = self.bottleneck_translator(x)

        x = self.wavelet_recon1(x, skip1)
        x = self.translator2(x, tskip1)

        x = self.conv_out(x)

        x = rearrange(x, 'b c t h w -> b t c h w')
        return x




if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"


    model = WaST_level1(in_shape=(4, 2, 32, 32), encoder_dim=20, block_list=[2, 8, 2]).cuda()
    print(model)
    dummy_tensor = torch.rand(1, 4, 2, 32, 32).cuda()
    output = model(dummy_tensor)
    print(f"input shape is {dummy_tensor.shape}, output shape is {output.shape}...")
    flops = FlopCountAnalysis(model, dummy_tensor)
    print(flop_count_table(flops))






