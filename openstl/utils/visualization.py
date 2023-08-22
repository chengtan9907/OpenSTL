from typing import Union

import cv2
import os
import imageio
import numpy as np

import matplotlib.pyplot as plt


def imshow(img: Union[str, np.ndarray],
           win_name: str = '',
           wait_time: int = 0):
    """Show an image.

    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    if isinstance(img, str):
        img = cv2.imread(img)
    assert isinstance(img, np.ndarray)
    cv2.imshow(win_name, img)
    if wait_time == 0:  # prevent from hanging if windows was closed
        while True:
            ret = cv2.waitKey(1)
            closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
            # if user closed window or if some key pressed
            if closed or ret != -1:
                break
    else:
        ret = cv2.waitKey(wait_time)


def plot_world_map(out_path=None):
    """plot the world map with Basemap"""
    try:
        from mpl_toolkits.basemap import Basemap
    except:
        assert False and 'Please install Basemap, e.g., pip install geos basemap pyproj'

    fig = plt.figure(figsize=(8, 4))
    fig.add_axes([0., 0., 1, 1])
    map = Basemap()
    map.drawcoastlines(linewidth=2)
    map.drawcountries(linewidth=1)
    plt.show()
    if out_path is not None:
        plt.savefig(out_path, dpi=300, format='png')
    plt.close()


def get_mpl_colormap(cmap_name):
    """mapping matplotlib cmap to cv2"""
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:,2::-1]

    return color_range.reshape(256, 1, 3)


def show_video_line(data, ncols, vmax=0.6, vmin=0.0, cmap='gray', norm=None, cbar=False, format='png', out_path=None, use_rgb=False):
    """generate images with a video sequence"""
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(3.25 * ncols, 3))
    plt.subplots_adjust(wspace=0.01, hspace=0)

    if len(data.shape) > 3:
        data = data.swapaxes(1,2).swapaxes(2,3)

    images = []
    if ncols == 1:
        if use_rgb:
            im = axes.imshow(cv2.cvtColor(data[0], cv2.COLOR_BGR2RGB))
        else:
            im = axes.imshow(data[0], cmap=cmap, norm=norm)
        images.append(im)
        axes.axis('off')
        im.set_clim(vmin, vmax)
    else:
        for t, ax in enumerate(axes.flat):
            if use_rgb:
                im = ax.imshow(cv2.cvtColor(data[t], cv2.COLOR_BGR2RGB), cmap='gray')
            else:
                im = ax.imshow(data[t], cmap=cmap, norm=norm)
            images.append(im)
            ax.axis('off')
            im.set_clim(vmin, vmax)

    if cbar and ncols > 1:
        cbaxes = fig.add_axes([0.9, 0.15, 0.04 / ncols, 0.7]) 
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.1, cax=cbaxes)

    plt.show()
    if out_path is not None:
        fig.savefig(out_path, format=format, pad_inches=0, bbox_inches='tight')
    plt.close()


def show_video_gif_multiple(prev, true, pred, vmax=0.6, vmin=0.0, cmap='gray', norm=None, out_path=None, use_rgb=False):
    """generate gif with a video sequence"""

    def swap_axes(x):
        if len(x.shape) > 3:
            return x.swapaxes(1,2).swapaxes(2,3)
        else: return x

    prev, true, pred = map(swap_axes, [prev, true, pred])
    prev_frames = prev.shape[0]
    frames = prev_frames + true.shape[0]
    images = []
    for i in range(frames):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 6))
        for t, ax in enumerate(axes):
            if t == 0:
                plt.text(0.3, 1.05, 'ground truth', fontsize=15, color='green', transform=ax.transAxes)
                if i < prev_frames:
                    if use_rgb:
                        im = ax.imshow(cv2.cvtColor(prev[i], cv2.COLOR_BGR2RGB))
                    else:
                        im = ax.imshow(prev[i], cmap=cmap, norm=norm)
                else:
                    if use_rgb:
                        im = ax.imshow(cv2.cvtColor(true[i-frames], cv2.COLOR_BGR2RGB))
                    else:
                        im = ax.imshow(true[i-frames], cmap=cmap, norm=norm)
            elif t == 1:
                plt.text(0.2, 1.05, 'predicted frames', fontsize=15, color='red', transform=ax.transAxes)
                if i < prev_frames:
                    if use_rgb:
                        im = ax.imshow(cv2.cvtColor(prev[i], cv2.COLOR_BGR2RGB))
                    else:
                        im = ax.imshow(prev[i], cmap=cmap, norm=norm)
                else:
                    if use_rgb:
                        im = ax.imshow(cv2.cvtColor(pred[i-frames], cv2.COLOR_BGR2RGB))
                    else:
                        im = ax.imshow(pred[i-frames], cmap=cmap, norm=norm)
            ax.axis('off')
            im.set_clim(vmin, vmax)
        plt.savefig('./tmp.png', bbox_inches='tight', format='png')
        images.append(imageio.imread('./tmp.png'))
    plt.close()
    os.remove('./tmp.png')

    if out_path is not None:
        if not out_path.endswith('gif'):
            out_path = out_path + '.gif'
        imageio.mimsave(out_path, images)


def show_video_gif_single(data, out_path=None, use_rgb=False):
    """generate gif with a video sequence"""
    images = []
    if len(data.shape) > 3:
        data=data.swapaxes(1, 2).swapaxes(2, 3)

    images = []
    for i in range(data.shape[0]):
        if use_rgb:
            data[i] = cv2.cvtColor(data[i], cv2.COLOR_BGR2RGB)
        image = imageio.core.util.Array(data[i])
        images.append(image)

    if out_path is not None:
        if not out_path.endswith('gif'):
            out_path = out_path + '.gif'
        imageio.mimsave(out_path, images)


def show_heatmap_on_image(img: np.ndarray,
                          mask: np.ndarray,
                          use_rgb: bool = False,
                          colormap: int = cv2.COLORMAP_JET,
                          image_weight: float = 0.5,
                          image_binary: bool = False) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

        img: The base image in RGB or BGR format.
        mask: The cam mask.
        use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
        colormap: The OpenCV colormap to be used.
        image_weight: The final result is image_weight * img + (1-image_weight) * mask.
        image_binary: Whether to binarize the image.

    returns: The default image with the cam overlay.
    """
    if mask.shape[0] != img.shape[1]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1]. Got: {image_weight}")

    if not image_binary:
        cam = (1 - image_weight) * heatmap + image_weight * img
    else:
        cam = (1 - image_weight) * heatmap + image_weight * img
        mask = 255 * img[:, :, 0] < 100.
        cam[mask, :] = img[mask, :]
    cam = cam / np.max(cam)

    return np.uint8(255 * cam)


def show_taxibj(heatmap, cmap='viridis', title=None, out_path=None, vis_channel=None):
    """ploting heatmap to show or save of TaxiBJ"""
    if vis_channel is not None:
        vis_channel = 0 if vis_channel < 0 else vis_channel
    else:
        vis_channel = 0

    cmap = get_mpl_colormap(cmap)
    ret_img = list()
    if len(heatmap.shape) == 3:
        heatmap = heatmap[np.newaxis, :]

    for i in range(heatmap.shape[0]):
        # plot heatmap with cmap
        vis_img = heatmap[i, vis_channel, :, :, np.newaxis]
        vis_img = cv2.resize(np.uint8(255 * vis_img), (256, 256)).squeeze()
        vis_img = cv2.applyColorMap(np.uint8(vis_img), cmap)
        vis_img = np.float32(vis_img) / 255
        vis_img = vis_img / np.max(vis_img)
        vis_img = np.uint8(255 * vis_img)

        ret_img.append(vis_img[np.newaxis, :])
        if out_path is not None:
            cv2.imwrite(str(out_path).replace('.', f'{i}.'), vis_img)
        if title is not None:
            imshow(vis_img, win_name=title+str(i))

    if len(ret_img) > 1:
        return np.concatenate(ret_img, axis=0)
    else:
        return ret_img[0]


def show_weather_bench(heatmap, src_img=None, cmap='GnBu', title=None,
                       out_path=None, vis_channel=None):
    """fusing src_img and heatmap to show or save of Weather Bench"""
    if not isinstance(src_img, np.ndarray):
        if src_img is None:
            plot_world_map('tmp.png')
            src_img = cv2.imread('tmp.png')
            os.remove('./tmp.png')
        elif isinstance(src_img, str):
            src_img = cv2.imread(src_img)
        src_img = cv2.resize(src_img, (512, 256))
    src_img = np.float32(src_img) / 255
    if vis_channel is not None:
        vis_channel = 0 if vis_channel < 0 else vis_channel
    else:
        vis_channel = 0

    ret_img = list()
    if len(heatmap.shape) == 3:
        heatmap = heatmap[np.newaxis, :]

    for i in range(heatmap.shape[0]):
        vis_img = show_heatmap_on_image(
            src_img, heatmap[i, vis_channel, ...], use_rgb=False, colormap=get_mpl_colormap(cmap),
            image_weight=0.1, image_binary=True)
        ret_img.append(vis_img[np.newaxis, :])
        if out_path is not None:
            cv2.imwrite(str(out_path).replace('.', f'{i}.'), vis_img)
        if title is not None:
            imshow(vis_img, win_name=title+str(i))

    if len(ret_img) > 1:
        return np.concatenate(ret_img, axis=0)
    else:
        return ret_img[0]
