#!/usr/bin/env bash

# run this script on the root
mkdir -p data/kth
cd data/kth

FORMAT='jpg'

# download the image format `kth_action.zip`
if [ $FORMAT == 'jpg' ]; then
    wget https://github.com/chengtan9907/OpenSTL/releases/download/v0.1.0/kth_action.zip -O kth_action.zip
    unzip kth_action.zip
fi

# download all kth datasets from `https://www.csc.kth.se/cvap/actions/` as `.avi` format, which
#   should be converted to the image format.
if [ $FORMAT == 'avi' ]; then
    wget http://www.csc.kth.se/cvap/actions/walking.zip -O walking.zip
    unzip walking.zip
    wget http://www.csc.kth.se/cvap/actions/jogging.zip -O jogging.zip
    unzip jogging.zip
    wget http://www.csc.kth.se/cvap/actions/running.zip -O running.zip
    unzip running.zip
    wget http://www.csc.kth.se/cvap/actions/boxing.zip -O boxing.zip
    unzip boxing.zip
    wget http://www.csc.kth.se/cvap/actions/handwaving.zip -O handwaving.zip
    unzip handwaving.zip
    wget http://www.csc.kth.se/cvap/actions/handclapping.zip -O handclapping.zip
    unzip handclapping.zip
fi

rm *.zip

echo "finished"


# Download and arrange them in the following structure:
# OpenSTL
# └── data
#     ├── kth
#     │   ├── boxing
#     │   ├── handclapping
#     │   ├── handwaving
#     │   ├── jogging
#     │   ├── running
#     │   ├── walking
