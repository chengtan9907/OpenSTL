# Adding new datasets

## Locations to add hardcoded parameters for new dataset

/configs/newdatasetname ***done***

make new directory in /data

/simvp/datasets/\_\_init__.py  ***done***
* import custom dataloader
* add dataloader name to \_\_all__

/simvp/datasets/dataloader.py ***done***
* add elif 
* returns train, validation, and test dataloaders for dataset.
    * expected return from \_\_getitem__ method of dataset class is a pytorch tensor with dimensions (timesteps, channels, height, width), where timesteps is the total number of steps including input and output

/simvp/datasets/dataset_constant.py ***done***
* add to dataset_parameters
    * defines in_shape and sequence lengths


# Adding new models

/simvp/utils/parser.py add to --method argument choices