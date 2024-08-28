#!/bin/bash
python GeoSeg/tools/create_masks.py --state "SH" --epsg "EPSG:25832"
python GeoSeg/tools/create_masks.py --state "NRW_1" --epsg "EPSG:25832"
python GeoSeg/tools/create_masks.py --state "NRW_3" --epsg "EPSG:25832"

python GeoSeg/tools/copy_files_transferability.py

python GeoSeg/tools/tof_patch_split.py \
--img-dir "data/combination_1/train_images" \
--mask-dir "data/combination_1/train_masks" \
--output-img-dir "data/combination_1/train/images_1024" \
--output-mask-dir "data/combination_1/train/masks_1024"\
--mode "train" --split-size 1024 --stride 1024 \

python GeoSeg/tools/tof_patch_split.py \
--img-dir "data/combination_1/test_images" \
--mask-dir "data/combination_1/test_masks" \
--output-img-dir "data/combination_1/test/images_5000" \
--output-mask-dir "data/combination_1/test/masks_5000" \
--mode "val" --split-size 5000 --stride 5000 \

python GeoSeg/tools/tof_patch_split.py \
--img-dir "data/combination_1/val_images" \
--mask-dir "data/combination_1/val_masks" \
--output-img-dir "data/combination_1/val/images_1024" \
--output-mask-dir "data/combination_1/val/masks_1024" \
--mode "val" --split-size 1024 --stride 1024 \

python GeoSeg/tools/tof_patch_split.py \
--img-dir "data/combination_2/train_images" \
--mask-dir "data/combination_2/train_masks" \
--output-img-dir "data/combination_2/train/images_1024" \
--output-mask-dir "data/combination_2/train/masks_1024"\
--mode "train" --split-size 1024 --stride 1024 \

python GeoSeg/tools/tof_patch_split.py \
--img-dir "data/combination_2/test_images" \
--mask-dir "data/combination_2/test_masks" \
--output-img-dir "data/combination_2/test/images_5000" \
--output-mask-dir "data/combination_2/test/masks_5000" \
--mode "val" --split-size 5000 --stride 5000 \

python GeoSeg/tools/tof_patch_split.py \
--img-dir "data/combination_2/val_images" \
--mask-dir "data/combination_2/val_masks" \
--output-img-dir "data/combination_2/val/images_1024" \
--output-mask-dir "data/combination_2/val/masks_1024" \
--mode "val" --split-size 1024 --stride 1024 \

python GeoSeg/tools/tof_patch_split.py \
--img-dir "data/combination_3/train_images" \
--mask-dir "data/combination_3/train_masks" \
--output-img-dir "data/combination_3/train/images_1024" \
--output-mask-dir "data/combination_3/train/masks_1024"\
--mode "train" --split-size 1024 --stride 1024 \

python GeoSeg/tools/tof_patch_split.py \
--img-dir "data/combination_3/test_images" \
--mask-dir "data/combination_3/test_masks" \
--output-img-dir "data/combination_3/test/images_5000" \
--output-mask-dir "data/combination_3/test/masks_5000" \
--mode "val" --split-size 5000 --stride 5000 \

python GeoSeg/tools/tof_patch_split.py \
--img-dir "data/combination_3/val_images" \
--mask-dir "data/combination_3/val_masks" \
--output-img-dir "data/combination_3/val/images_1024" \
--output-mask-dir "data/combination_3/val/masks_1024" \
--mode "val" --split-size 1024 --stride 1024 \

python GeoSeg/tools/tof_patch_split.py \
--img-dir "data/combination_4/train_images" \
--mask-dir "data/combination_4/train_masks" \
--output-img-dir "data/combination_4/train/images_1024" \
--output-mask-dir "data/combination_4/train/masks_1024"\
--mode "train" --split-size 1024 --stride 1024 \

python GeoSeg/tools/tof_patch_split.py \
--img-dir "data/combination_4/test_images" \
--mask-dir "data/combination_4/test_masks" \
--output-img-dir "data/combination_4/test/images_5000" \
--output-mask-dir "data/combination_4/test/masks_5000" \
--mode "val" --split-size 5000 --stride 5000 \

python GeoSeg/tools/tof_patch_split.py \
--img-dir "data/combination_4/val_images" \
--mask-dir "data/combination_4/val_masks" \
--output-img-dir "data/combination_4/val/images_1024" \
--output-mask-dir "data/combination_4/val/masks_1024" \
--mode "val" --split-size 1024 --stride 1024 \

python GeoSeg/train_supervision.py -c GeoSeg/config/tof_transfer/ftunetformer_com1.py
python GeoSeg/train_supervision.py -c GeoSeg/config/tof_transfer/ftunetformer_com2.py
python GeoSeg/train_supervision.py -c GeoSeg/config/tof_transfer/ftunetformer_com3.py
python GeoSeg/train_supervision.py -c GeoSeg/config/tof_transfer/ftunetformer_com4.py