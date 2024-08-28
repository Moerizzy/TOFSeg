#!/bin/bash

python GeoSeg/tof_test.py -c GeoSeg/config/tof/unet.py -o fig_results/tof/unet
python GeoSeg/tools/stitch_together.py "fig_results/tof/unet"
python GeoSeg/tof_test.py -c GeoSeg/config/tif_tof/unet.py -o fig_results/tif_tof/unet
python GeoSeg/tof_test.py -c GeoSeg/config/tif_tof/unetformer.py -o fig_results/tif_tof/unetformer
python GeoSeg/tools/stitch_together.py "fig_results/tif_tof/unetformer"
python GeoSeg/tof_test.py -c GeoSeg/config/tif_tof/dcswin_small.py -o fig_results/tif_tof/dcswin_small
python GeoSeg/tools/stitch_together.py "fig_results/tif_tof/dcswin_small"
python GeoSeg/tof_test.py -c GeoSeg/config/tif_tof/dcswin_tiny.py -o fig_results/tif_tof/dcswin_tiny
python GeoSeg/tools/stitch_together.py "fig_results/tif_tof/dcswin_tiny"
python GeoSeg/tof_test.py -c GeoSeg/config/tif_tof/dcswin_base.py -o fig_results/tif_tof/dcswin_base
python GeoSeg/tools/stitch_together.py "fig_results/tif_tof/dcswin_base"
python GeoSeg/tof_test.py -c GeoSeg/config/tif_tof/ftunetformer.py -o fig_results/tif_tof/ftunetformer
python GeoSeg/tools/stitch_together.py "fig_results/tif_tof/ftunetformer"
python GeoSeg/tof_test.py -c GeoSeg/config/tif_tof/banet.py -o fig_results/tif_tof/banet
python GeoSeg/tools/stitch_together.py "fig_results/tif_tof/banet"
python GeoSeg/tof_test.py -c GeoSeg/config/tif_tof/abcnet.py -o fig_results/tif_tof/abcnet
python GeoSeg/tools/stitch_together.py "fig_results/tif_tof/abcnet"
python GeoSeg/tof_test.py -c GeoSeg/config/tif_tof/a2fpn.py -o fig_results/tif_tof/a2fpn
python GeoSeg/tools/stitch_together.py "fig_results/tif_tof/a2fpn"
python GeoSeg/tof_test.py -c GeoSeg/config/tif_tof/unetformer_lsk_s.py -o fig_results/tif_tof/unetformer_lsk_s
python GeoSeg/tools/stitch_together.py "fig_results/tif_tof/unetformer_lsk_s"
python GeoSeg/tof_test.py -c GeoSeg/config/tif_tof/unetformer_lsk_t.py -o fig_results/tif_tof/unetformer_lsk_t
python GeoSeg/tools/stitch_together.py "fig_results/tif_tof/unetformer_lsk_t"