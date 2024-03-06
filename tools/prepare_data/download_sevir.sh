#!/usr/bin/env bash

#run this script on the root
mkdir -p data/sevir

# download origin sevir and place them in 'data/sevir/'


echo "finished download origin sevir dataset" 

# convert ori_sevir to dataloader_sevir
mkdir -p data/sevir/processed
# # convert sevir_vil
# python tools/prepare_data/generate_sevir.py --sevir_data data/sevir --data_name vil --output_dir data/sevir/processed
# # convert sevir_vis
# python tools/prepare_data/generate_sevir.py --sevir_data data/sevir --data_name vis --output_dir data/sevir/processed
# # convert sevir_ir069
# python tools/prepare_data/generate_sevir.py --sevir_data data/sevir --data_name ir069 --output_dir data/sevir/processed
# convert sevir_ir107
python tools/prepare_data/generate_sevir.py --sevir_data data/sevir --data_name ir107 --output_dir data/sevir/processed

echo "finished"

# Download and arrange them in the following structure:
# OpenSTL
# ©¸©¤©¤data
#    ©À©¤©¤ sevir
#    ©¦   ©À©¤©¤ ir069
#    ©¦   ©À©¤©¤ ir107
#    ©¦   ©À©¤©¤ vis
#    ©¦   ©À©¤©¤ vil
#    ©¦   ©À©¤©¤ lght
#    ©¦   ©À©¤©¤ processed
#    ©¦   ©¦   ©À©¤©¤ ir069_training.h5
#    ©¦   ©¦   ©À©¤©¤ ir069_testing.h5
#    ©¦   ©¦   ©À©¤©¤ ir107_training.h5
#    ©¦   ©¦   ©À©¤©¤ ir107_testing.h5
#    ©¦   ©¦   ©À©¤©¤ vis_training.h5
#    ©¦   ©¦   ©À©¤©¤ vis_testing.h5
#    ©¦   ©¦   ©À©¤©¤ vil_training.h5
#    ©¦   ©¦   ©À©¤©¤ vil_testing.h5
