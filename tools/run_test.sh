#!/bin/bash

python GeoSeg/tof_test.py -c GeoSeg/config/tof/dcswin.py -o fig_results/tof_final/dcswin -t "d4"
python GeoSeg/tof_test.py -c GeoSeg/config/tof/ftunetformer.py -o fig_results/tof_final/ftunetformer -t "d4"
python GeoSeg/tof_test.py -c GeoSeg/config/tof/banet.py -o fig_results/tof_final/banet -t "d4"
python GeoSeg/tof_test.py -c GeoSeg/config/tof/a2fpn.py -o fig_results/tof_final/a2fpn -t "d4"