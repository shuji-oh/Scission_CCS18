#!/bin/sh

if [ $# -ne 1 ]; then
    echo "Please input sampling rate as an argument." 1>&2
    exit 1
fi

python3 create_labels.py > voltage_$1MS.csv

ls -w 1 20211023-rpi_red/ | xargs -I '{}' python3 extract_all.py '20211023-rpi_red/{}' $1 ECU1 >> voltage_$1MS.csv

ls -w 1 20211022-ard1/ | xargs -I '{}' python3 extract_all.py '20211022-ard1/{}' $1 ECU2 >> voltage_$1MS.csv

ls -w 1 20211022-ard2/ | xargs -I '{}' python3 extract_all.py '20211022-ard2/{}' $1 ECU3 >> voltage_$1MS.csv

ls -w 1 20211022-suzuki_ecu/ | xargs -I '{}' python3 extract_all.py '20211022-suzuki_ecu/{}' $1 ECU4 >> voltage_$1MS.csv  

ls -w 1 20211025-rpi_yellow/ | xargs -I '{}' python3 extract_all.py '20211025-rpi_yellow/{}' $1 ECU5 >> voltage_$1MS.csv

ls -w 1 20211022-suzuki_meter/ | xargs -I '{}' python3 extract_all.py '20211022-suzuki_meter/{}' $1 ECU6 >> voltage_$1MS.csv
