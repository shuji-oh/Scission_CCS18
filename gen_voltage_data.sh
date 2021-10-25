#!/bin/sh

python3 create_labels.py > voltage.csv

ls -w 1 20211023-rpi_red/ | xargs -I '{}' python3 extract_all.py '20211023-rpi_red/{}' ECU1 >> voltage.csv

ls -w 1 20211022-ard1/ | xargs -I '{}' python3 extract_all.py '20211022-ard1/{}' ECU2 >> voltage.csv

ls -w 1 20211022-ard2/ | xargs -I '{}' python3 extract_all.py '20211022-ard2/{}' ECU3 >> voltage.csv

ls -w 1 20211022-suzuki_ecu/ | xargs -I '{}' python3 extract_all.py '20211022-suzuki_ecu/{}' ECU4 >> voltage.csv  

ls -w 1 20211025-rpi_yellow/ | xargs -I '{}' python3 extract_all.py '20211023-rpi_yellow/{}' ECU5 >> voltage.csv

ls -w 1 20211022-suzuki_meter/ | xargs -I '{}' python3 extract_all.py '20211022-suzuki_meter/{}' ECU6 >> voltage.csv
