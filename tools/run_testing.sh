#!/bin/bash

python GeoSeg/tof_test.py -c GeoSeg/config/tof/unet.py -o fig_results/tof/unet
python GeoSeg/tools/stitch_together.py "fig_results/tof/unet"
# python GeoSeg/tof_test.py -c GeoSeg/config/tof/unetformer.py -o fig_results/tof/unetformer
# python GeoSeg/tools/stitch_together.py "fig_results/tof/unetformer"
# python GeoSeg/tof_test.py -c GeoSeg/config/tof/dcswin_small.py -o fig_results/tof/dcswin_small
# python GeoSeg/tools/stitch_together.py "fig_results/tof/dcswin_small"
# python GeoSeg/tof_test.py -c GeoSeg/config/tof/dcswin_tiny.py -o fig_results/tof/dcswin_tiny
# python GeoSeg/tools/stitch_together.py "fig_results/tof/dcswin_tiny"
# python GeoSeg/tof_test.py -c GeoSeg/config/tof/dcswin_base.py -o fig_results/tof/dcswin_base
# python GeoSeg/tools/stitch_together.py "fig_results/tof/dcswin_base"
python GeoSeg/tof_test.py -c GeoSeg/config/tof/ftunetformer.py -o fig_results/tof/ftunetformer
python GeoSeg/tools/stitch_together.py "fig_results/tof/ftunetformer"
# python GeoSeg/tof_test.py -c GeoSeg/config/tof/banet.py -o fig_results/tof/banet
# python GeoSeg/tools/stitch_together.py "fig_results/tof/banet"
# python GeoSeg/tof_test.py -c GeoSeg/config/tof/abcnet.py -o fig_results/tof/abcnet
# python GeoSeg/tools/stitch_together.py "fig_results/tof/abcnet"
# python GeoSeg/tof_test.py -c GeoSeg/config/tof/a2fpn.py -o fig_results/tof/a2fpn
# python GeoSeg/tools/stitch_together.py "fig_results/tof/a2fpn"
# python GeoSeg/tof_test.py -c GeoSeg/config/tof/unetformer_lsk_s.py -o fig_results/tof/unetformer_lsk_s
# python GeoSeg/tools/stitch_together.py "fig_results/tof/unetformer_lsk_s"
# python GeoSeg/tof_test.py -c GeoSeg/config/tof/unetformer_lsk_t.py -o fig_results/tof/unetformer_lsk_t
# python GeoSeg/tools/stitch_together.py "fig_results/tof/unetformer_lsk_t"