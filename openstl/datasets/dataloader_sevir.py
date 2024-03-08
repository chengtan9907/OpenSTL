"""
Input generator for sevir
"""

import os
import os.path as osp
import numpy as np
import pandas as pd
import h5py
import torch
import datetime

from openstl.datasets.utils import create_loader
os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'

from torch.utils.data import Dataset


# Nominal Frame time offsets in minutes (used for non-raster types)

# NOTE:  The lightning flashes in each from will represent the 5 minutes leading up the
# the frame's time EXCEPT for the first frame, which will use the same flashes as the second frame
#  (This will be corrected in a future version of SEVIR so that all frames are consistent)
FRAME_TIMES = np.arange(-120.0,125.0,5) * 60 # in seconds

# Record dtypes for reading
DTYPES={'vil':np.uint8,'vis':np.int16,'ir069':np.int16,'ir107':np.int16,'lght':np.int16}

class SEVIRDataset(Dataset):
    """
    Sequence class for generating batches from SEVIR
    
    Parameters
    ----------
    batch_size  int
       batch size to generate
    n_batch_per_epoch  int or None
       Number of batches in an epoch.  Set to None to match available data
    unwrap_time   bool
       If True, single images are returned instead of image sequences
    shuffle  bool
       If True, data samples are shuffled before each epoch
    shuffle_seed   int
       Seed to use for shuffling
    output_type  np.dtype
       dtype of generated tensors
    normalize_x  list of tuple
       list the same size as x_img_types containing tuples (scale,offset) used to 
       normalize data via   X  -->  (X-offset)*scale.  If None, no scaling is done
    normalize_y  list of tuple
       list the same size as y_img_types containing tuples (scale,offset) used to 
       normalize data via   X  -->  (X-offset)*scale
    
    Returns
    -------
    SEVIRDataset generator
    
    Examples
    --------
    
        # Get just Radar image sequences
        vil_seq = SEVIRDataset(x_img_types=['vil'],batch_size=16)
        X = vil_seq.__getitem__(1234)  # returns list the same size as x_img_types passed to constructor
        
        # Get ir satellite+lightning as X,  radar for Y
        vil_ir_lght_seq = SEVIRDataset(x_img_types=['ir107','lght'],y_img_types=['vil'],batch_size=4)
        X,Y = vil_ir_lght_seq.__getitem__(420)  # X,Y are lists same length as x_img_types and y_img_types
        
        # Get single images of VIL
        vil_imgs = SEVIRDataset(x_img_types=['vil'], batch_size=256, unwrap_time=True, shuffle=True)
        
        # Filter out some times
        vis_seq = SEVIRDataset(x_img_types=['vis'],batch_size=32,unwrap_time=True,
                                start_date=datetime.datetime(2018,1,1),
                                end_date=datetime.datetime(2019,1,1),
                                datetime_filter=lambda t: np.logical_and(t.dt.hour>=13,t.dt.hour<=21))
    
    """
    def __init__(self,
                 data_root='data/sevir/processed',
                 data_name='vil',
                 is_train=True,
                 use_augment=False,
                 n_batch_per_epoch=None,
                 unwrap_time=False,
                 shuffle=False,
                 shuffle_seed=1,
                 output_type=np.float32,
                 normalize_x=None,
                 normalize_y=None,
                 ):
        self._samples = {}
        self.is_train = is_train
        if is_train:
            self._hdf_file = osp.join(data_root, f'{data_name}_training.h5')
        else:
            self._hdf_file = osp.join(data_root, f'{data_name}_testing.h5')
        self.n_batch_per_epoch = n_batch_per_epoch

        self.use_augment = use_augment
        self.unwrap_time = unwrap_time
        self.shuffle=shuffle
        self.shuffle_seed=int(shuffle_seed)
        self.output_type=output_type
        self.normalize_x = normalize_x
        self.normalize_y = normalize_y
        self.data_name = data_name
        
        self.mean = 33.44
        self.std = 47.54
        self._read_samples()
    

    def on_epoch_end(self):
        if self.shuffle:
            self._samples.sample(frac=1,random_state=self.shuffle_seed)
    
    def close(self):
        """
        Closes all open file handles
        """
        self.hf.close()
        self._hdf_file=str()

    def __del__(self):
        self.hf.close()

    def __len__(self):
        """
        How many batches to generate per epoch
        """
        if self._samples is not None:
            max_n = self._samples['IN']._dset.shape[0]
        else:
            max_n = 0
        if self.n_batch_per_epoch is not None:
            return min(self.n_batch_per_epoch, max_n)
        else:
            return max_n

    def _augment_seq(self, X, Y):
        pass
    
    def _to_tensor(self, X, Y):
        return torch.from_numpy(X), torch.from_numpy(Y)

        
    def __getitem__(self, idx):
        """
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            resize(imread(file_name), (200, 200))
               for file_name in batch_x]), np.array(batch_y)    
        """
        X, Y = self._get_batch_samples(idx)
        
        #augmentation
        if self.use_augment:
            X, Y = self._augment_seq(X, Y)
        else:
            X, Y = self._to_tensor(X, Y)
        if self.normalize_x:
            X = SEVIRDataset.normalize(X,self.normalize_x)
        else:
            X = (X - self.mean) / self.std

        if self.normalize_y:
            Y = SEVIRDataset.normalize(Y,self.normalize_y)
        else:
            Y = (Y - self.mean) / self.std
        
        #transform
        X = X.unsqueeze(dim=0).permute(3, 0, 1, 2)
        Y = Y.unsqueeze(dim=0).permute(3, 0, 1, 2)
        return X, Y
        
    def _get_batch_samples(self,idx):
        return self._samples['IN'][idx], self._samples['OUT'][idx]


    def _lght_to_grid(self,data,t_slice=slice(0,None)):
        """
        Converts Nx5 lightning data matrix into a 2D grid of pixel counts
        """
        #out_size = (48,48,len(FRAME_TIMES)-1) if isinstance(t_slice,(slice,)) else (48,48)
        out_size = (48,48,len(FRAME_TIMES)) if t_slice.stop is None else (48,48,1)
        if data.shape[0]==0:
            return np.zeros((1,)+out_size,dtype=np.float32)
        
        # filter out points outside the grid
        x,y=data[:,3],data[:,4]
        m=np.logical_and.reduce( [x>=0,x<out_size[0],y>=0,y<out_size[1]] )
        data=data[m,:]
        if data.shape[0]==0:
            return np.zeros((1,)+out_size,dtype=np.float32)
        
        # Filter/separate times
        t=data[:,0]
        if t_slice.stop is not None:  # select only one time bin
            if t_slice.stop>0:
                if t_slice.stop < len(FRAME_TIMES):
                    tm=np.logical_and( t>=FRAME_TIMES[t_slice.stop-1],
                                       t< FRAME_TIMES[t_slice.stop] )
                else:
                    tm=t>=FRAME_TIMES[-1]
            else: # special case:  frame 0 uses lght from frame 1
                tm=np.logical_and( t>=FRAME_TIMES[0],t<FRAME_TIMES[1] )
            #tm=np.logical_and( (t>=FRAME_TIMES[t_slice],t<FRAME_TIMES[t_slice+1]) )
      
            data=data[tm,:]
            z=np.zeros( data.shape[0], dtype=np.int64 )
        else: # compute z coodinate based on bin locaiton times
            z=np.digitize(t,FRAME_TIMES)-1
            z[z==-1]=0 # special case:  frame 0 uses lght from frame 1
           
        x=data[:,3].astype(np.int64)
        y=data[:,4].astype(np.int64)
        
        k=np.ravel_multi_index(np.array([y,x,z]),out_size)
        n = np.bincount(k,minlength=np.prod(out_size))
        return np.reshape(n,out_size).astype(np.int16)[np.newaxis,:]
         
    def _read_samples(self):
        """
        Read samples from converted h5 file
        """
        hf = h5py.File(self._hdf_file, mode='r')
        self.hf = hf
        IN = hf['IN']
        OUT = hf['OUT']
        assert len(IN) == len(OUT), f'{self._hdf_file} has different number of input and output'
        IN = IN.astype(self.output_type)
        OUT = OUT.astype(self.output_type)
        self._samples['IN'] = IN
        self._samples['OUT'] = OUT

 
    
    @staticmethod
    def normalize(X,s):
        """
        Normalized data using s = (scale,offset) via Z = (X-offset)*scale
        """
        return (X-s[1])*s[0]

    @staticmethod
    def unnormalize(Z,s):
        """
        Reverses the normalization performed in a SEVIRDataset generator
        given s=(scale,offset)
        """
        return Z/s[0]+s[1]
     

def load_data(batch_size,
              val_batch_size,
              data_root='./data/sevir/',
              num_workers = 4,
              data_name='vil',
              in_shape=[13, 1, 192, 192],
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False,
              **kwargs):
    data_root = osp.join(data_root, 'sevir', 'processed')
    train_set = SEVIRDataset(data_root, data_name, is_train=True, use_augment=use_augment,
                             output_type=DTYPES[data_name])
    test_set = SEVIRDataset(data_root, data_name, is_train=False, use_augment=use_augment,
                            output_type=DTYPES[data_name])

    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers,
                                     distributed=distributed, 
                                     use_prefetcher=use_prefetcher)
    dataloader_vali = create_loader(test_set, # validation_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)

    return dataloader_train, dataloader_vali, dataloader_test

if __name__ == '__main__':
    path = './data/sevir'
    dataloader_train, dataloader_vali, dataloader_test = load_data(8,8)
    for item in dataloader_train:
        x,y=item
