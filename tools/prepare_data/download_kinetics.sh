#!/usr/bin/env bash

# run this script on the root of OpenSTL,
# we provide this script according to https://github.com/cvdfoundation/kinetics-dataset
mkdir -p data
cd data


# *** Step 1: run https://github.com/cvdfoundation/kinetics-dataset/k400_downloader.sh
# Download directories vars
root_dl="kinetics400"
root_dl_targz="k400_targz"

# Make root directories
[ ! -d $root_dl ] && mkdir $root_dl
[ ! -d $root_dl_targz ] && mkdir $root_dl_targz

# Download train tars, will resume
curr_dl=${root_dl_targz}/train
url=https://s3.amazonaws.com/kinetics/400/train/k400_train_path.txt
[ ! -d $curr_dl ] && mkdir -p $curr_dl
wget -c -i $url -P $curr_dl

# Download validation tars, will resume
curr_dl=${root_dl_targz}/val
url=https://s3.amazonaws.com/kinetics/400/val/k400_val_path.txt
[ ! -d $curr_dl ] && mkdir -p $curr_dl
wget -c -i $url -P $curr_dl

# Download test tars, will resume
curr_dl=${root_dl_targz}/test
url=https://s3.amazonaws.com/kinetics/400/test/k400_test_path.txt
[ ! -d $curr_dl ] && mkdir -p $curr_dl
wget -c -i $url -P $curr_dl

# Download replacement tars, will resume
curr_dl=${root_dl_targz}/replacement
url=https://s3.amazonaws.com/kinetics/400/replacement_for_corrupted_k400.tgz
[ ! -d $curr_dl ] && mkdir -p $curr_dl
wget -c $url -P $curr_dl

# Download annotations csv files
curr_dl=${root_dl}/annotations
url_tr=https://s3.amazonaws.com/kinetics/400/annotations/train.csv
url_v=https://s3.amazonaws.com/kinetics/400/annotations/val.csv
url_t=https://s3.amazonaws.com/kinetics/400/annotations/test.csv
[ ! -d $curr_dl ] && mkdir -p $curr_dl
wget -c $url_tr -P $curr_dl
wget -c $url_v -P $curr_dl
wget -c $url_t -P $curr_dl

# Download readme
url=http://s3.amazonaws.com/kinetics/400/readme.md
wget -c $url -P $root_dl

# Downloads complete
echo -e "\nDownloads complete! Now run extractor, k400_extractor.sh"


# *** Step 2: run https://github.com/cvdfoundation/kinetics-dataset/k400_extractor.sh
# Download directories vars
# root_dl="kinetics400"
# root_dl_targz="k400_targz"

# Make root directories
[ ! -d $root_dl ] && mkdir $root_dl

# Extract train
curr_dl=$root_dl_targz/train
curr_extract=$root_dl/train
[ ! -d $curr_extract ] && mkdir -p $curr_extract
tar_list=$(ls $curr_dl)
for f in $tar_list
do
	[[ $f == *.tar.gz ]] && echo Extracting $curr_dl/$f to $curr_extract && tar zxf $curr_dl/$f -C $curr_extract
done

# Extract validation
curr_dl=$root_dl_targz/val
curr_extract=$root_dl/val
[ ! -d $curr_extract ] && mkdir -p $curr_extract
tar_list=$(ls $curr_dl)
for f in $tar_list
do
	[[ $f == *.tar.gz ]] && echo Extracting $curr_dl/$f to $curr_extract && tar zxf $curr_dl/$f -C $curr_extract
done

# Extract test
curr_dl=$root_dl_targz/test
curr_extract=$root_dl/test
[ ! -d $curr_extract ] && mkdir -p $curr_extract
tar_list=$(ls $curr_dl)
for f in $tar_list
do
	[[ $f == *.tar.gz ]] && echo Extracting $curr_dl/$f to $curr_extract && tar zxf $curr_dl/$f -C $curr_extract
done

# Extract replacement
curr_dl=$root_dl_targz/replacement
curr_extract=$root_dl/replacement
[ ! -d $curr_extract ] && mkdir -p $curr_extract
tar_list=$(ls $curr_dl)
for f in $tar_list
do
	[[ $f == *.tgz ]] && echo Extracting $curr_dl/$f to $curr_extract && tar zxf $curr_dl/$f -C $curr_extract
done

# Extraction complete
echo -e "\nExtractions complete!"


# Download and arrange them in the following structure:
# OpenSTL
# └── data
#     ├── kinetics400
#     │   ├── annotations
#     │   ├── replacement
#     │   ├── test
#     │   ├── train
#     │   ├── val
