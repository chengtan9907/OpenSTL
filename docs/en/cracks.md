# SimVP2 Documentation

## Call flow

<br>
<br>
<br>
<br>
<br>
<br>
<br>

python tools/non_dist_train.py -d cracks --lr 1e-3 -c ./configs/cracks/SimVP.py --ex_name cracks_1


## Adding new dataset

### /configs/newdatasetname ***done***

<br>

### make new directory in /data

<br>

### /simvp/datasets/\_\_init__.py  ***done***
* import custom dataloader
* add dataloader name to \_\_all__

<br>

### /simvp/datasets/dataloader.py ***done***
* add elif 
* returns train, validation, and test dataloaders for dataset.
    * expected return from \_\_getitem__ method of dataset class is a pytorch tensor with dimensions (timesteps, channels, height, width), where timesteps is the total number of steps including input and output

### /simvp/datasets/dataset_constant.py ***done***
* add to dataset_parameters
    * defines in_shape and sequence lengths

<br>
<br>
<br>
<br>
<br>

# Adding new models

/simvp/utils/parser.py add to --method argument choices