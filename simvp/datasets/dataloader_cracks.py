from PIL import Image
import torch
from torch.utils.data import Dataset
from .video_dataset import VideoFolder, video_loader, make_dataset, find_videos
import torchvision.transforms as transforms


def load_data(batch_size, val_batch_size, data_root, num_workers=4, pre_seq_length=5, aft_seq_length=1):
    img_width = 128
    validation_batch_size = 1
    num_workers = 4
    train_set_path = './data/cracks/train'
    test_set_path = './data/cracks/test'

    transform_video = transforms.Compose([transforms.Resize(
        size=(128, 128), interpolation=Image.NEAREST), transforms.ToTensor(),])

    train_set = Cracks(video_root=train_set_path, video_ext='png',
                            nframes=6, loader=video_loader, transform=transform_video)
    test_set = Cracks(video_root=test_set_path, video_ext='png',
                           nframes=6, loader=video_loader, transform=transform_video)
    

    dataloader_train = torch.utils.data.DataLoader(train_set,
                                                   batch_size=batch_size, shuffle=True,
                                                   pin_memory=True, drop_last=True,
                                                   num_workers=num_workers)
    dataloader_vali = None
                        # torch.utils.data.DataLoader(test_set,
                        #                           batch_size=val_batch_size, shuffle=False,
                        #                           pin_memory=True, drop_last=True,
                        #                           num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(test_set,
                                                  batch_size=val_batch_size, shuffle=False,
                                                  pin_memory=True, drop_last=True,
                                                  num_workers=num_workers)
    
    return dataloader_train, dataloader_vali, dataloader_test



class Cracks(VideoFolder):

    # copying from taxibj to see essential methods

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
        self.mean = 0
        self.std = 1

# below is load data from taxibj


'''
def load_data(batch_size, val_batch_size, data_root,
              num_workers=4, pre_seq_length=None, aft_seq_length=None):

    dataset = np.load(data_root+'taxibj/dataset.npz')
    X_train, Y_train, X_test, Y_test = dataset['X_train'], dataset[
        'Y_train'], dataset['X_test'], dataset['Y_test']
    train_set = TaxibjDataset(X=X_train, Y=Y_train)
    test_set = TaxibjDataset(X=X_test, Y=Y_test)

    dataloader_train = torch.utils.data.DataLoader(train_set,
                                                   batch_size=batch_size, shuffle=True,
                                                   pin_memory=True, drop_last=True,
                                                   num_workers=num_workers)
    dataloader_vali = torch.utils.data.DataLoader(test_set,
                                                  batch_size=val_batch_size, shuffle=False,
                                                  pin_memory=True, drop_last=True,
                                                  num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(test_set,
                                                  batch_size=val_batch_size, shuffle=False,
                                                  pin_memory=True, drop_last=True,
                                                  num_workers=num_workers)

    return dataloader_train, dataloader_vali, dataloader_test

'''
