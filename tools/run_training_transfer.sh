#!/bin/bash

python GeoSeg/train_supervision.py -c GeoSeg/config/tof_transfer/ftunetformer_com1.py
python GeoSeg/train_supervision.py -c GeoSeg/config/tof_transfer/ftunetformer_com2.py
python GeoSeg/train_supervision.py -c GeoSeg/config/tof_transfer/ftunetformer_com3.py
python GeoSeg/train_supervision.py -c GeoSeg/config/tof_transfer/ftunetformer_com4.py