#!/usr/bin/env bash

# run this script on the root
mkdir -p data/human
cd data/human

# # Download H36M annotations
# wget http://visiondata.cis.upenn.edu/volumetric/h36m/h36m_annot.tar
# tar -xf h36m_annot.tar
# # rm h36m_annot.tar

# Download H36M images (original resolutions 1024x1024x3)
mkdir images_source
cd images_source
wget http://visiondata.cis.upenn.edu/volumetric/h36m/S1.tar
tar -xf S1.tar
rm S1.tar
wget http://visiondata.cis.upenn.edu/volumetric/h36m/S5.tar
tar -xf S5.tar
rm S5.tar
wget http://visiondata.cis.upenn.edu/volumetric/h36m/S6.tar
tar -xf S6.tar
rm S6.tar
wget http://visiondata.cis.upenn.edu/volumetric/h36m/S7.tar
tar -xf S7.tar
rm S7.tar
wget http://visiondata.cis.upenn.edu/volumetric/h36m/S8.tar
tar -xf S8.tar
rm S8.tar
wget http://visiondata.cis.upenn.edu/volumetric/h36m/S9.tar
tar -xf S9.tar
rm S9.tar
wget http://visiondata.cis.upenn.edu/volumetric/h36m/S11.tar
tar -xf S11.tar
rm S11.tar
cd ../../..

# Preprocess the original images to get the low resolution 256x256 (can also be downloaded from Baidu Cloud)
python tools/prepare_data/resize_image.py data/human --src_name images_source --dst_name images --shape 256
rm -r data/human/images_source

# Prepare the meta files, provide by `https://github.com/ZhengChang467/STRPM`
cp tools/prepare_data/meta/human/test.txt data/human/
cp tools/prepare_data/meta/human/train.txt data/human/

echo "finished"


# Download and arrange them in the following structure:
# OpenSTL
# └── data
#     ├── human
#     │   ├── images
#     │   ├── test.txt
#     │   ├── train.txt
