import cv2
import numpy as np
import torch

try:
    import lpips
    from skimage.metrics import structural_similarity as cal_ssim
except:
    lpips = None
    cal_ssim = None


def rescale(x):
    return (x - x.max()) / (x.max() - x.min()) * 2 - 1


def MAE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean(np.abs(pred-true), axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean(np.abs(pred-true) / norm, axis=(0, 1)).sum()


def MSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean((pred-true)**2, axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean((pred-true)**2 / norm, axis=(0, 1)).sum()


def RMSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.sqrt(np.mean((pred-true)**2, axis=(0, 1)).sum())
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.sqrt(np.mean((pred-true)**2 / norm, axis=(0, 1)).sum())


# cite the `PSNR` code from E3d-LSTM, Thanks!
# https://github.com/google/e3d_lstm/blob/master/src/trainer.py
def PSNR(pred, true):
    mse = np.mean((np.uint8(pred * 255)-np.uint8(true * 255))**2)
    return 20 * np.log10(255) - 10 * np.log10(mse)


def SSIM(pred, true, **kwargs):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = pred.astype(np.float64)
    img2 = true.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


class LPIPS(torch.nn.Module):
    """Learned Perceptual Image Patch Similarity, LPIPS.

    Modified from
    https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py
    """

    def __init__(self, net='alex', use_gpu=True):
        super().__init__()
        assert net in ['alex', 'squeeze', 'vgg']
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.loss_fn = lpips.LPIPS(net=net)
        if use_gpu:
            self.loss_fn.cuda()

    def forward(self, img1, img2):
        # Load images, which are min-max norm to [0, 1]
        img1 = lpips.im2tensor(img1 * 255)  # RGB image from [-1,1]
        img2 = lpips.im2tensor(img2 * 255)
        if self.use_gpu:
            img1, img2 = img1.cuda(), img2.cuda()
        return self.loss_fn.forward(img1, img2).squeeze().detach().cpu().numpy()


def metric(pred, true, mean=None, std=None, metrics=['mae', 'mse'],
           clip_range=[0, 1], channel_names=None,
           spatial_norm=False, return_log=True):
    """The evaluation function to output metrics.

    Args:
        pred (tensor): The prediction values of output prediction.
        true (tensor): The prediction values of output prediction.
        mean (tensor): The mean of the preprocessed video data.
        std (tensor): The std of the preprocessed video data.
        metric (str | list[str]): Metrics to be evaluated.
        clip_range (list): Range of prediction to prevent overflow.
        channel_names (list | None): The name of different channels.
        spatial_norm (bool): Weather to normalize the metric by HxW.
        return_log (bool): Whether to return the log string.

    Returns:
        dict: evaluation results
    """
    if mean is not None and std is not None:
        pred = pred * std + mean
        true = true * std + mean
    eval_res = {}
    eval_log = ""
    allowed_metrics = ['mae', 'mse', 'rmse', 'ssim', 'psnr', 'lpips']
    invalid_metrics = set(metrics) - set(allowed_metrics)
    if len(invalid_metrics) != 0:
        raise ValueError(f'metric {invalid_metrics} is not supported.')
    if isinstance(channel_names, list):
        assert pred.shape[2] % len(channel_names) == 0 and len(channel_names) > 1
        c_group = len(channel_names)
        c_width = pred.shape[2] // c_group
    else:
        channel_names, c_group, c_width = None, None, None

    if 'mse' in metrics:
        if channel_names is None:
            eval_res['mse'] = MSE(pred, true, spatial_norm)
        else:
            mse_sum = 0.
            for i, c_name in enumerate(channel_names):
                eval_res[f'mse_{str(c_name)}'] = MSE(pred[:, :, i*c_width: (i+1)*c_width, ...],
                                                     true[:, :, i*c_width: (i+1)*c_width, ...], spatial_norm)
                mse_sum += eval_res[f'mse_{str(c_name)}']
            eval_res['mse'] = mse_sum / c_group

    if 'mae' in metrics:
        if channel_names is None:
            eval_res['mae'] = MAE(pred, true, spatial_norm)
        else:
            mae_sum = 0.
            for i, c_name in enumerate(channel_names):
                eval_res[f'mae_{str(c_name)}'] = MAE(pred[:, :, i*c_width: (i+1)*c_width, ...],
                                                     true[:, :, i*c_width: (i+1)*c_width, ...], spatial_norm)
                mae_sum += eval_res[f'mae_{str(c_name)}']
            eval_res['mae'] = mae_sum / c_group

    if 'rmse' in metrics:
        if channel_names is None:
            eval_res['rmse'] = RMSE(pred, true, spatial_norm)
        else:
            rmse_sum = 0.
            for i, c_name in enumerate(channel_names):
                eval_res[f'rmse_{str(c_name)}'] = RMSE(pred[:, :, i*c_width: (i+1)*c_width, ...],
                                                       true[:, :, i*c_width: (i+1)*c_width, ...], spatial_norm)
                rmse_sum += eval_res[f'rmse_{str(c_name)}']
            eval_res['rmse'] = rmse_sum / c_group

    pred = np.maximum(pred, clip_range[0])
    pred = np.minimum(pred, clip_range[1])
    if 'ssim' in metrics:
        ssim = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                ssim += cal_ssim(pred[b, f].swapaxes(0, 2),
                                 true[b, f].swapaxes(0, 2), multichannel=True)
        eval_res['ssim'] = ssim / (pred.shape[0] * pred.shape[1])

    if 'psnr' in metrics:
        psnr = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                psnr += PSNR(pred[b, f], true[b, f])
        eval_res['psnr'] = psnr / (pred.shape[0] * pred.shape[1])

    if 'lpips' in metrics:
        lpips = 0
        cal_lpips = LPIPS(net='alex', use_gpu=False)
        pred = pred.transpose(0, 1, 3, 4, 2)
        true = true.transpose(0, 1, 3, 4, 2)
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                lpips += cal_lpips(pred[b, f], true[b, f])
        eval_res['lpips'] = lpips / (pred.shape[0] * pred.shape[1])

    if return_log:
        for k, v in eval_res.items():
            eval_str = f"{k}:{v}" if len(eval_log) == 0 else f", {k}:{v}"
            eval_log += eval_str

    return eval_res, eval_log
