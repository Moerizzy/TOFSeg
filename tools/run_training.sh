#!/bin/bash

#_
python GeoSeg/train_supervision.py -c GeoSeg/config/tof/dcswin.py
python GeoSeg/train_supervision.py -c GeoSeg/config/tof/ftunetformer.py
python GeoSeg/train_supervision.py -c GeoSeg/config/tof/banet.py
python GeoSeg/train_supervision.py -c GeoSeg/config/tof/abcnet.py
python GeoSeg/train_supervision.py -c GeoSeg/config/tof/a2fpn.py
python GeoSeg/train_supervision.py -c GeoSeg/config/tof/unetformer_lsk_s.py
python GeoSeg/train_supervision.py -c GeoSeg/config/tof/unetformer_lsk_t.py