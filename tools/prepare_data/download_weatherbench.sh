#!/usr/bin/env bash

# run this script on the root
mkdir data/weather
cd data/weather

# down 5.625deg `2m_temperature` (32x64) and place them in `data/weather/` according to `https://github.com/pangeo-data/WeatherBench`
wget "https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg%2F2m_temperature&files=2m_temperature_5.625deg.zip" -O 2m_temperature_5.625deg.zip

# down 1.40625deg `2m_temperature` (128x256) and place them in `data/weather/` according to `https://github.com/pangeo-data/WeatherBench`
# wget "https://dataserv.ub.tum.de/s/m1524895/download?path=%2F1.40625deg%2F2m_temperature&files=2m_temperature_1.40625deg.zip" -O 2m_temperature_1.40625deg.zip

# download and arrange them in the following structure:
# SimVPv2
# └── data
#     ├── weather
#     │   ├── 2m_temperature
#     │   ├── ...
