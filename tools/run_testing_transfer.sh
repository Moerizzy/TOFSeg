#!/bin/bash

python GeoSeg/tof_test.py -c GeoSeg/config/tof_transfer/ftunetformer_com1.py -o fig_results/tof_transfer/ftunetformer_com1
python GeoSeg/tools/stitch_together.py "fig_results/tof/ftunetformer_com1"

python GeoSeg/tof_test.py -c GeoSeg/config/tof_transfer/ftunetformer_com2.py -o fig_results/tof_transfer/ftunetformer_com2
python GeoSeg/tools/stitch_together.py "fig_results/tof/ftunetformer_com2"

python GeoSeg/tof_test.py -c GeoSeg/config/tof_transfer/ftunetformer_com3.py -o fig_results/tof_transfer/ftunetformer_com3
python GeoSeg/tools/stitch_together.py "fig_results/tof/ftunetformer_com3"

python GeoSeg/tof_test.py -c GeoSeg/config/tof_transfer/ftunetformer_com4.py -o fig_results/tof_transfer/ftunetformer_com4
python GeoSeg/tools/stitch_together.py "fig_results/tof/ftunetformer_com4"