import glob
import os
import numpy as np
import cv2
from PIL import Image
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-dir", default="data/tof/train_masks")
    parser.add_argument("--output-dir", default="data/tof/statistics")
    return parser.parse_args()

def calc_stats():
    

masks_dir = args.mask_dir

mask_paths_raw = glob.glob(os.path.join(masks_dir, "*.tif"))

mask = Image.open(mask_path).convert("RGB")

bins = np.array(range(num_classes + 1))
class_pixel_counts, _ = np.histogram(mask_crop, bins=bins)
cf = class_pixel_counts / (mask_crop.shape[0] * mask_crop.shape[1])

t0 = time.time()
mpp.Pool(processes=mp.cpu_count()).map(calc_stats, inp)
t1 = time.time()

    split_time = t1 - t0
    print("images spliting spends: {} s".format(split_time))
