#!/bin/bash
python GeoSeg/train_supervision.py -c GeoSeg/config/tof/unet.py
python GeoSeg/train_supervision.py -c GeoSeg/config/tif_tof/unet.py
python GeoSeg/train_supervision.py -c GeoSeg/config/tif_tof/unetformer.py
python GeoSeg/train_supervision.py -c GeoSeg/config/tif_tof/dcswin.py
python GeoSeg/train_supervision.py -c GeoSeg/config/tif_tof/dcswin_tiny.py
python GeoSeg/train_supervision.py -c GeoSeg/config/tif_tof/dcswin_base.py
python GeoSeg/train_supervision.py -c GeoSeg/config/tif_tof/ftunetformer.py
python GeoSeg/train_supervision.py -c GeoSeg/config/tif_tof/banet.py
python GeoSeg/train_supervision.py -c GeoSeg/config/tif_tof/abcnet.py
python GeoSeg/train_supervision.py -c GeoSeg/config/tif_tof/a2fpn.py
python GeoSeg/train_supervision.py -c GeoSeg/config/tif_tof/unetformer_lsk_s.py
python GeoSeg/train_supervision.py -c GeoSeg/config/tif_tof/unetformer_lsk_t.py