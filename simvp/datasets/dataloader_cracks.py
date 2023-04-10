from PIL import Image
import torch
from torch.utils.data import Dataset
from .video_dataset import VideoFolder, video_loader
import torchvision.transforms as transforms


def load_data(batch_size, val_batch_size, data_root, num_workers=4, pre_seq_length=5, aft_seq_length=1):
    img_width = 128
    validation_batch_size = 1
    num_workers = 4
    train_set_path = './data/cracks/train'
    test_set_path = './data/cracks/test'

    transform_video = transforms.Compose([transforms.Resize(
        size=(128, 128), interpolation=Image.NEAREST), transforms.ToTensor(),])

    train_set = VideoFolder(video_root=train_set_path, video_ext='png',
                            nframes=6, loader=video_loader, transform=transform_video)
    test_set = VideoFolder(video_root=test_set_path, video_ext='png',
                           nframes=6, loader=video_loader, transform=transform_video)

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=torch.Generator(device='cpu'))
#            train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    dataloader_validation = None

    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, generator=torch.Generator(device='cpu'))
#            test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader_train, dataloader_validation, dataloader_test


class Cracks(Dataset):

    # copying from taxibj to see essential methods

    def __init__(self, X, Y):
        super(Cracks, self).__init__()
        self.X = (X+1)/2
        self.Y = (Y+1)/2
        self.mean = 0
        self.std = 1

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.X[index, ::]).float()
        labels = torch.tensor(self.Y[index, ::]).float()
        return data, labels
    

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
