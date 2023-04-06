# =============================================================================
# Load Data
# =============================================================================

import os
import math
from PIL import Image
import torch
import torch.utils.data as data


def has_file_allowed_extension(filename, ext):
    '''
    Checks if a file extension is an allowed extension.
    
    Args:
        filename: Path to file. (str)
        ext: List of allowed extensions. (list[str])
        
    Returns:
        bool: True if the filename ends with an allowed extension. (bool)
        
    ---------------------------------------------------------------------------
    code reference: https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
    '''
    
    filename_lower = filename.lower()
    return any(filename_lower.endswith(i) for i in ext)


def pil_loader(path):
    '''
    ---------------------------------------------------------------------------
    code reference: https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
    '''
    
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            if len(img.getbands())==1:
                return img.convert('L')
            else:
                return img.convert('RGB')


def accimage_loader(path):
    '''
    ---------------------------------------------------------------------------
    code reference: https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
    '''
    
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

    
def default_image_loader():
    '''
    ---------------------------------------------------------------------------
    code reference: https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
    '''
    
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def find_videos(video_root):
    '''
    This function finds subdirectories containing the full length videos in the root video folder. 
    
    Args: 
        video_root: Path to root directory of video folders. (str)
            
    Returns: 
        video_names: List of video names. (list)
        video_idx: Dict with items (video_names, video_idx). (dict)
        
    ---------------------------------------------------------------------------
    code adapted from: https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py        
    '''
    
    video_names = [d for d in os.listdir(video_root) if os.path.isdir(os.path.join(video_root, d))]
    video_names.sort()
    video_idx = {video_names[i]: i for i in range(len(video_names))}
    return video_names, video_idx


def video_loader(frame_paths):
    '''
    Function to load a sequence of video frames given a list of their paths.
    
    Args: 
        sample_paths: List of paths to video frames to make a video sample. (list[str])
    
    Returns: 
        video: List of video frames. (list[Image])
    '''
 
    image_loader = default_image_loader()
    video = []
    for path in frame_paths:
        frame_path = path
        if os.path.exists(path):
            video.append(image_loader(frame_path))
        else:
            return video
    return video


def make_dataset(video_root, ext, nframes):
    '''
    Function to create a video dataset, where each sample sequence contains 'nframes' frames.
    
    Args: 
        video_root: Path to root directory of video folders. (str)
        ext: List of allowed extensions. (list[str])
        nframes: Number of frames per video sequence. (int)
    
    Returns: 
        dataset: List of lists of tuples (video_dir (str), video_idx (int)) (list)
                 (List of samples, where each sample is a list of nframe number
                 of frame lists (tuple: (video_dir (str), video_idx (int)))
    '''
    
    # get video folders and assign index to each video
    video_names, video_idx = find_videos(video_root)
    
    frames = []
    dataset = []
    
    for i in range(len(video_names)):   
        video_dir = os.path.join(video_root, video_names[i])
        
        # get directories and names of files in video folder
        for file_root, _, file_names in sorted(os.walk(video_dir)):
            
            # get number of frames in video folder to make samples of nframes number of frames
            nsample_files = math.floor(len(file_names)/nframes)*nframes
            
            for file_name in sorted(file_names)[0:nsample_files]:
                # check whether file extension of file in video folder is an allowed extension
                if has_file_allowed_extension(file_name, ext):
                    file_path = os.path.join(file_root, file_name)            
                    # make a list of all frames in video folder
                    frame = (file_path, video_idx[video_names[i]])
                    frames.append(frame)
                    
    # make a list of samples (samples are lists of nframe number of frames from video folder) 
    for j in range(0, (len(frames)-nframes), nframes):
        sample = frames[j:j+nframes]
        dataset.append(sample)

    return dataset


class VideoFolder(data.Dataset):
    '''
    Class to create a dataset of sequences (length nframes) of video frames.   
    
    Data is assumed to be arranged in this way:
        video_root/video/frame.ext -> subset/video1/frame1.ext
                                   -> subset/video1/frame2.ext
                                   -> subset/video2/frame1.ext
        
    Args:   
        video_root: Path to root directory of video folders. (str)
        video_ext: List of allowed extensions. (list[str])
        nframes: Number of frames per video sequence. (int)
        loader: Function to load a sample given its path. (callable)
        transform: Function to perform transformations on video frames (callable, optional)
    '''
    
    def __init__(self, video_root, video_ext, nframes, loader=video_loader, transform=None):
        video_names, video_idx = find_videos(video_root)
        video_dataset = make_dataset(video_root, video_ext, nframes)
        
        self.root = video_root
        self.ext = video_ext
        self.nframes = nframes
        self.loader = loader
        
        self.video_names = video_names
        self.video_idx = video_idx
        self.video_dataset = video_dataset

        self.transform = transform
        
        
    def __getitem__(self, idx):
        '''
        Args:   
            idx: Index of dataset sample. (int)
        
        Returns: 
            video_sample: torch.FloatTensor containing one video sample of nframes
                          with pixel range [-1,1]. (torch.FloatTensor)
                          Shape of torch.FloatTensor: [C,D,H,W].
                          (C: nimg_channels, D: nframes, H: img_h, W: img_w)
        '''
                
        frame_paths = []
        sample_video_idx = []        
        
        # get frame paths and video_idx of samples in dataset
        for frame_path, frame_video_idx in self.video_dataset[idx][:]:
            frame_paths.append(frame_path)
            sample_video_idx.append(frame_video_idx)
        
        # load frames of the video sample into a list of images
        video_sample = self.loader(frame_paths)
        
        # transform image
        if self.transform is not None:
#            video_sample = [self.transform(frame) for frame in video_sample]            
            # and make pixel range [-1,1]
            video_sample = [self.transform(frame).mul(2).add(-1) for frame in video_sample]
            
        # make torch.FloatTensor form video sample
        # video_sample = torch.stack(video_sample, 0).permute(1,0,2,3)
        video_sample = torch.stack(video_sample, 0)
        initial_frames, target_frame = video_sample.split([5, 1], dim=0)
        return initial_frames, target_frame


    def __len__(self):
        return len(self.video_dataset)
