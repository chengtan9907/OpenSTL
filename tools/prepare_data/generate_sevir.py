"""
Input generator for sevir
"""

import argparse
import os
import os.path as osp
import logging
import numpy as np
import pandas as pd
import h5py
import torch
import datetime

os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'

from torch.utils.data import Dataset

# List all avaialbe types
TYPES    = ['vis','ir069','ir107','vil','lght']

import pathlib
_thisdir = str(pathlib.Path(__file__).parent.absolute())
DEFAULT_CATALOG   = _thisdir+'/../CATALOG.csv'
DEFAULT_DATA_HOME = _thisdir+'/../data'

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
    catalog  str or pd.DataFrame
        name of SEVIR catalog file to be read in, or an already read in and processed catalog
    x_img_types  list 
        List of image types to be used as model inputs.  For types, run SEVIRDataset.get_types()
    y_img_types  list or None
       List of image types to be used as model targets (if None, __getitem__ returns only x_img_types )
    sevir_data_home  str
       Directory path to SEVIR data
    catalog  str
       Name of SEVIR catalog CSV file.  
    batch_size  int
       batch size to generate
    n_batch_per_epoch  int or None
       Number of batches in an epoch.  Set to None to match available data
    start_date   datetime
       Start time of SEVIR samples to generate   
    end_date    datetime
       End time of SEVIR samples to generate
    datetime_filter   function
       Mask function applied to time_utc column of catalog (return true to keep the row). 
       Pass function of the form   lambda t : COND(t)
       Example:  lambda t: np.logical_and(t.dt.hour>=13,t.dt.hour<=21)  # Generate only day-time events
    catalog_filter  function
       Mask function applied to entire catalog dataframe (return true to keep row).  
       Pass function of the form lambda catalog:  COND(catalog)
       Example:  lambda c:  [s[0]=='S' for s in c.id]   # Generate only the 'S' events
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
                 x_img_types=['vil'],
                 y_img_types=None, 
                 catalog=DEFAULT_CATALOG,
                 batch_size = 3,
                 n_batch_per_epoch=None,
                 start_date=None,
                 end_date=None,
                 datetime_filter=None,
                 catalog_filter=None,
                 unwrap_time=False,
                 sevir_data_home=DEFAULT_DATA_HOME,
                 shuffle=False,
                 shuffle_seed=1,
                 output_type=np.float32,
                 normalize_x=None,
                 normalize_y=None,
                 verbose=False,
                 data_name='vil'
                 ):
        self._samples = None
        self._hdf_files = {}
        self.x_img_types = x_img_types
        self.y_img_types = y_img_types
        if isinstance(catalog, (str,)):
            self.catalog=pd.read_csv(catalog, parse_dates=['time_utc'], low_memory=False)
        else:
            self.catalog=catalog
        self.batch_size=batch_size
        self.n_batch_per_epoch = n_batch_per_epoch

        self.datetime_filter=datetime_filter
        self.catalog_filter=catalog_filter
        self.start_date=start_date
        self.end_date=end_date
        self.unwrap_time = unwrap_time
        self.sevir_data_home=sevir_data_home
        self.shuffle=shuffle
        self.shuffle_seed=int(shuffle_seed)
        self.output_type=output_type
        self.normalize_x = normalize_x
        self.normalize_y = normalize_y
        self.verbose=verbose
        self.data_name = data_name
        if normalize_x:
            assert(len(normalize_x)==len(x_img_types))
        if normalize_y:
            assert(len(normalize_y)==len(y_img_types))

        if self.start_date:
            self.catalog = self.catalog[self.catalog.time_utc > self.start_date ]
        if self.end_date:
            self.catalog = self.catalog[self.catalog.time_utc <= self.end_date]
        if self.datetime_filter:
            self.catalog = self.catalog[self.datetime_filter(self.catalog.time_utc)]
        
        if self.catalog_filter:
            self.catalog = self.catalog[self.catalog_filter(self.catalog)]
        
        self._compute_samples()
        self._open_files(verbose=self.verbose)
    
    def load_batches(self,
                     n_batches=10,
                     offset=0,
                     progress_bar=False):
        """
        Loads a selected number of batches into memory.  This returns the concatenated
        result of [self.__getitem__(i+offset) for i in range(n_batches)]

        WARNING:  Be careful about running out of memory.

        Parameters
        ----------
        n_batches   int
            Number of batches to load.   Set to -1 to load them all, but becareful
            not to run out of memory
        offset int
            batch offset to apply
        progress_bar  bool
            Show a progress bar during loading (requires tqdm module)

        """
        if progress_bar:
            try:
                from tqdm import tqdm as RW
            except ImportError:
                print('You need to install tqdm to use progress bar')
                RW=list
        else:
            RW=list
        
        n_batches = self.__len__() if n_batches==-1 else n_batches
        n_batches = min(n_batches,self.__len__())
        assert(n_batches>0)
        
        def out_shape(n_batches,shp,batch_size):
            """
            Computes shape for preinitialization
            """
            return (n_batches*batch_size,*shp)

        bidx=0
        if self.y_img_types is None: # one output
            X = None
            for i in RW( range(offset,offset+n_batches) ):
                Xi = self.__getitem__(i)
                if X is None:
                    shps = [out_shape(n_batches,xi.shape[1:],xi.shape[0]) for xi in Xi] 
                    X = [np.empty( s,dtype=DTYPES[k] ) for s,k in zip(shps,self.x_img_types)]
                for ii,xi in enumerate(Xi):
                    X[ii][bidx:bidx+xi.shape[0]] = xi
                bidx+=xi.shape[0]
            return X
        else:
            X,Y=None,None
            for i in RW( range(offset,offset+n_batches) ):
                Xi,Yi = self.__getitem__(i)
                if X is None:
                    shps_x = [out_shape(n_batches,xi.shape[1:],xi.shape[0]) for xi in Xi]
                    shps_y = [out_shape(n_batches,yi.shape[1:],yi.shape[0]) for yi in Yi]
                    X = [np.empty(s,dtype=DTYPES[k]) for s,k in zip(shps_x,self.x_img_types)]
                    Y = [np.empty(s,dtype=DTYPES[k]) for s,k in zip(shps_y,self.y_img_types)]
                for ii,xi in enumerate(Xi):
                    X[ii][bidx:bidx+xi.shape[0]] = xi
                for ii,yi in enumerate(Yi):
                    Y[ii][bidx:bidx+yi.shape[0]] = yi   
                bidx+=xi.shape[0]
            return X,Y

    def on_epoch_end(self):
        if self.shuffle:
            self._samples.sample(frac=1,random_state=self.shuffle_seed)
    
    def close(self):
        """
        Closes all open file handles
        """
        for f in self._hdf_files:
            self._hdf_files[f].close()
        self._hdf_files={}

    def __del__(self):
        for f,hf in self._hdf_files.items():
            try:
                hf.close()
            except ImportError:
                pass # okay when python shutting down


    def __len__(self):
        """
        How many batches to generate per epoch
        """
        if self._samples is not None:
            # Use floor to avoid sending a batch of < self.batch_size in last batch.   
            max_n = int(np.floor(self._samples.shape[0] / float(self.batch_size)))
        else:
            max_n = 0
        if self.n_batch_per_epoch is not None:
            return min(self.n_batch_per_epoch,max_n)
        else:
            return max_n
        
    def __getitem__(self, idx):
        """
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            resize(imread(file_name), (200, 200))
               for file_name in batch_x]), np.array(batch_y)    
        """
        batch = self._get_batch_samples(idx)
        data = {}
        for index, row in batch.iterrows():
            data = self._read_data(row,data)
        X = [data[t].astype(self.output_type) for t in self.x_img_types]

            
        if self.normalize_x:
            X = [SEVIRDataset.normalize(X[k],s) for k,s in enumerate(self.normalize_x)]

        if self.y_img_types is not None:
            Y = [data[t].astype(self.output_type) for t in self.y_img_types]
            if self.normalize_y:
                Y = [SEVIRDataset.normalize(Y[k],s) for k,s in enumerate(self.normalize_y)]
            
            return X,Y
        else:
            return X, None    
        
    def _get_batch_samples(self,idx):
        return self._samples.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
    
    def _read_data(self,row,data):
        """
        row is a series with fields IMGTYPE_filename, IMGTYPE_index, IMGTYPE_time_index
        """
        imgtyps = np.unique([x.split('_')[0] for x in list(row.keys())])
        for t in imgtyps:
            fname = row[f'{t}_filename']
            idx   = row[f'{t}_index']
            #t_slice = row[f'{t}_time_index'] if self.unwrap_time else slice(0,None)
            if self.unwrap_time:
                tidx=row[f'{t}_time_index']
                t_slice = slice(tidx,tidx+1) 
            else:
                t_slice = slice(0,None)
            # Need to bin lght counts into grid
            if t=='lght':
                lght_data = self._hdf_files[fname][idx][:]
                data_i = self._lght_to_grid(lght_data,t_slice)
            else:
                data_i = self._hdf_files[fname][t][idx:idx+1,:,:,t_slice]
            data[t] = np.concatenate( (data[t],data_i),axis=0 ) if (t in data) else data_i
            
        return data


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
         
    
    def _compute_samples(self):
        """
        Computes the list of samples in catalog to be used. This sets
           self._samples  

        """
        # locate all events containing colocated x_img_types and y_img_types
        imgt = self.x_img_types
        if self.y_img_types:
            imgt=list( set(imgt + self.y_img_types) ) # remove duplicates
        imgts = set(imgt)            
        filtcat = self.catalog[ np.logical_or.reduce([self.catalog.img_type==i for i in imgt]) ]
        # remove rows missing one or more requested img_types
        filtcat = filtcat.groupby('id').filter(lambda x: imgts.issubset(set(x['img_type'])))
        # If there are repeated IDs, remove them (this is a bug in SEVIR)
        filtcat = filtcat.groupby('id').filter(lambda x: x.shape[0]==len(imgt))
        self._samples = filtcat.groupby('id').apply( lambda df: self._df_to_series(df,imgt) )
        if self.shuffle:
            self._samples=self._samples.sample(frac=1,random_state=self.shuffle_seed)
        

    def _df_to_series(self,df,imgt):
        N_FRAMES=49  # TODO:  don't hardcode this
        d = {}
        df = df.set_index('img_type')
        for i in imgt:
            s = df.loc[i]
            idx = s.file_index if i!='lght' else s.id 
            if self.unwrap_time:
                d.update( {f'{i}_filename':[s.file_name]*N_FRAMES, 
                           f'{i}_index':[idx]*N_FRAMES,
                           f'{i}_time_index':range(N_FRAMES)} )   
            else:
                d.update( {f'{i}_filename':[s.file_name], 
                           f'{i}_index':[idx]} )
                   
        return pd.DataFrame(d)

    def _open_files(self,verbose=True):
        """
        Opens HDF files
        """
        imgt = self.x_img_types
        if self.y_img_types:
            imgt=list( set(imgt + self.y_img_types) ) # remove duplicates
        hdf_filenames = []
        for t in imgt:
            hdf_filenames += list(np.unique( self._samples[f'{t}_filename'].values ))
        self._hdf_files = {}
        for f in hdf_filenames:
            if verbose:
                print('Opening HDF5 file for reading',f)
            self._hdf_files[f] = h5py.File(self.sevir_data_home+'/'+f,'r') 
    
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
    
class NowcastGenerator(SEVIRDataset):
    """
    Generator that loads full VIL sequences, and spilts each
    event into three training samples, each 12 frames long.

    Event Frames:  [-----------------------------------------------]
                   [----13-----][---12----]
                               [----13----][----12----]
                                          [-----13----][----12----]
    """
    def __getitem__(self, idx):
        """

        """
        X,_ = super(NowcastGenerator, self).__getitem__(idx)  # N,L,W,49
        X = X[0]
        x1,x2,x3 = X[:, :, :, :13], X[:, :, :, 12:25], X[:, :, :, 24:37]
        y1,y2,y3 = X[:, :, :, 13:25], X[:, :, :, 25:37], X[:, :, :, 37:49]
        Xnew = np.concatenate((x1,x2,x3), axis=0)
        Ynew = np.concatenate((y1,y2,y3), axis=0)
        return [Xnew], [Ynew]
    

def get_nowcast_generator(sevir_location, batch_size=1, data_name='vil', start_date=None, end_date=None):
    sevir_catalog = osp.join(sevir_location, 'CATALOG.csv')
    filt = lambda c:  c.pct_missing==0 # remove samples with missing radar data
    return NowcastGenerator(catalog=sevir_catalog,
                            sevir_data_home=sevir_location,
                            x_img_types=[data_name],
                            y_img_types=[data_name],
                            data_name=data_name,
                            batch_size=batch_size,
                            start_date=start_date,
                            end_date=end_date,
                            catalog_filter=filt)

def collate_fn(batch):
    X, Y = [], []
    for item in batch:
        x, y = item
        X.append(x)
        Y.append(y)
    return torch.cat(X, dim = 0), torch.cat(Y, dim = 0)
    
def parser_args():
    parser = argparse.ArgumentParser(description='Make nowcast training & test datasets using SEVIR')
    parser.add_argument('--sevir_data', type=str, help='location of SEVIR dataset',default='data/sevir')
    parser.add_argument('--data_name', type=str, help='different sensor type for sevir data, choice from [vis, ir069, ir107, vil]')
    parser.add_argument('--output_dir', type=str, help='location of SEVIR dataset',default='../../data/interim')
    parser.add_argument('--n_chunks', type=int, help='Number of chucks to use (increase if memory limited)',default=20)

    args = parser.parse_args()
    return args

def read_write_chunks( filename, generator, n_chunks ):
    logger = logging.getLogger(__name__)
    chunksize = len(generator) // n_chunks
    # get first chunk
    logger.info(f'Gathering chunk 0/{n_chunks}')
    X,Y=generator.load_batches(n_batches=chunksize, offset=0, progress_bar=True)
    # Create datasets
    with h5py.File(filename, 'w') as hf:
      hf.create_dataset('IN', data=X[0],  maxshape=(None,X[0].shape[1],X[0].shape[2],X[0].shape[3]))
      hf.create_dataset('OUT', data=Y[0], maxshape=(None,Y[0].shape[1],Y[0].shape[2],Y[0].shape[3]))
    # Gather other chunks
    for c in range(1,n_chunks+1):
      offset = c*chunksize
      n_batches = min(chunksize,len(generator)-offset)
      if n_batches<0: # all done
        break
      logger.info('Gathering chunk %d/%s:' % (c,n_chunks))
      X,Y=generator.load_batches(n_batches=n_batches,offset=offset,progress_bar=True)
      with h5py.File(filename, 'a') as hf:
            hf['IN'].resize((hf['IN'].shape[0] + X[0].shape[0]), axis = 0)
            hf['OUT'].resize((hf['OUT'].shape[0] + Y[0].shape[0]), axis = 0)
            hf['IN'][-X[0].shape[0]:]  = X[0]
            hf['OUT'][-Y[0].shape[0]:] = Y[0]

def main(args):
    """ 
    Runs data processing scripts to extract training set from SEVIR
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    trn_generator = get_nowcast_generator(args.sevir_data, batch_size=8, data_name=args.data_name, end_date=datetime.datetime(2019,6,1))
    tst_generator = get_nowcast_generator(args.sevir_data, batch_size=8, data_name=args.data_name, start_date=datetime.datetime(2019,6,1))
    training_dir = osp.join(args.output_dir, f'{args.data_name}_training.h5') 
    logger.info(f'Reading/writing training data to {training_dir}')
    read_write_chunks(training_dir, trn_generator, args.n_chunks)
    testing_dir = osp.join(args.output_dir, f'{args.data_name}_testing.h5') 
    logger.info(f'Reading/writing testing data to {testing_dir}' )
    read_write_chunks(testing_dir, tst_generator, args.n_chunks)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    args = parser_args()
    main(args)




