from PIL import Image
import torch
from torch.utils.data import DataLoader
from .video_dataset import VideoFolder, video_loader
import torchvision.transforms as transforms

def load_data(batch_size, val_batch_size, data_root, num_workers=4, pre_seq_length=10, aft_seq_length=10):
    img_width=128
    validation_batch_size=1
    num_workers=4
    train_set_path='/home/haruka/datasets/FutureGAN_format/train'
    test_set_path='/home/haruka/datasets/FutureGAN_format/test'

    transform_video = transforms.Compose([transforms.Resize(size=(128, 128), interpolation=Image.NEAREST),transforms.ToTensor(),])

    train_set = VideoFolder(video_root=train_set_path, video_ext='png', nframes=6, loader=video_loader, transform=transform_video)
    test_set = VideoFolder(video_root=test_set_path, video_ext='png', nframes=6, loader=video_loader, transform=transform_video)

    dataloader_train = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=torch.Generator(device='cpu'))
#            train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader_validation = None
    dataloader_test = DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, generator=torch.Generator(device='cpu'))
#            test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader_train, dataloader_validation, dataloader_test
