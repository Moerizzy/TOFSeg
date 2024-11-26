import argparse
from pathlib import Path
import glob
from PIL import Image
import ttach as tta
import cv2
import numpy as np
import torch
import albumentations as albu
from catalyst.dl import SupervisedRunner
from skimage.morphology import remove_small_holes, remove_small_objects
from tools.cfg import py2cfg
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from train_supervision import *
import random
import os
import rasterio


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def building_to_rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [0, 0, 0]
    return mask_rgb


def pv2rgb(mask):  # Potsdam and vaihingen
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 204, 255]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 0, 255]
    return mask_rgb


def landcoverai_to_rgb(mask):
    w, h = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(w, h, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [233, 193, 133]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [0, 255, 0]
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb


def uavid2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [128, 0, 0]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [128, 64, 128]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [0, 128, 0]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [128, 128, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [64, 0, 128]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [192, 0, 192]
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [64, 64, 0]
    mask_rgb[np.all(mask_convert == 7, axis=0)] = [0, 0, 0]
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg(
        "-i",
        "--image_path",
        type=Path,
        required=True,
        help="Path to  huge image folder",
    )
    arg("-c", "--config_path", type=Path, required=True, help="Path to  config")
    arg(
        "-o",
        "--output_path",
        type=Path,
        help="Path to save resulting masks.",
        required=True,
    )
    arg(
        "-t",
        "--tta",
        help="Test time augmentation.",
        default=None,
        choices=[None, "d4", "lr"],
    )
    arg("-ph", "--patch-height", help="height of patch size", type=int, default=512)
    arg("-pw", "--patch-width", help="width of patch size", type=int, default=512)
    arg("-b", "--batch-size", help="batch size", type=int, default=2)
    arg(
        "-d",
        "--dataset",
        help="dataset",
        default="pv",
        choices=["pv", "landcoverai", "uavid", "building"],
    )
    return parser.parse_args()


def get_img_padded(image, patch_size):
    oh, ow = image.shape[0], image.shape[1]
    rh, rw = oh % patch_size[0], ow % patch_size[1]

    width_pad = 0 if rw == 0 else patch_size[1] - rw
    height_pad = 0 if rh == 0 else patch_size[0] - rh
    # print(oh, ow, rh, rw, height_pad, width_pad)
    h, w = oh + height_pad, ow + width_pad

    pad = albu.PadIfNeeded(
        min_height=h,
        min_width=w,
        position="bottom_right",
        border_mode=0,
        value=[0, 0, 0],
    )(image=image)
    img_pad = pad["image"]
    return img_pad, height_pad, width_pad


class InferenceDataset(Dataset):
    def __init__(self, tile_list=None, transform=albu.Normalize()):
        self.tile_list = tile_list
        self.transform = transform

    def __getitem__(self, index):
        img = self.tile_list[index]
        img_id = index
        aug = self.transform(image=img)
        img = aug["image"]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        results = dict(img_id=img_id, img=img)
        return results

    def __len__(self):
        return len(self.tile_list)


def make_dataset_for_one_huge_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    dataset = InferenceDataset(tile_list=[img])
    return (
        dataset,
        img.shape,
    )


def sliding_window_inference(model, image, num_classes, window_size=1024, stride=256):
    _, _, H, W = image.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    image = nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, padded_H, padded_W = image.shape

    # Initialize prediction tensor with the correct number of classes
    prediction = torch.zeros(
        (image.shape[0], num_classes, padded_H, padded_W), device=image.device
    )
    count = torch.zeros((1, 1, padded_H, padded_W), device=image.device)

    for h in range(0, padded_H - window_size + 1, stride):
        for w in range(0, padded_W - window_size + 1, stride):
            window = image[:, :, h : h + window_size, w : w + window_size]
            with torch.no_grad():
                output = model(window)
            prediction[:, :, h : h + window_size, w : w + window_size] += output
            count[:, :, h : h + window_size, w : w + window_size] += 1

    prediction /= count
    prediction = prediction[:, :, :H, :W]  # Remove padding
    return prediction


def main():
    args = get_args()
    seed_everything(42)
    patch_size = (args.patch_height, args.patch_width)
    config = py2cfg(args.config_path)
    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name + ".ckpt"),
        config=config,
    )

    model.cuda()
    model.eval()

    if args.tta == "lr":
        transforms = tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()])
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                # tta.VerticalFlip(),
                # tta.Rotate90(angles=[0, 90, 180, 270]),
                tta.Scale(scales=[0.75, 1, 1.25, 1.5, 1.75]),
                # tta.Multiply(factors=[0.8, 1, 1.2])
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)

    img_paths = []
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    for ext in ("*.tif", "*.png", "*.jpg"):
        img_paths.extend(glob.glob(os.path.join(args.image_path, ext)))
    img_paths.sort()
    # print(img_paths)
    for img_path in img_paths:
        img_name = img_path.split("/")[-1]
        # print('origin mask', original_mask.shape)
        (
            dataset,
            img_shape,
        ) = make_dataset_for_one_huge_image(img_path)

        output_height = img_shape[0]
        output_width = img_shape[1]

        output_mask = np.zeros(shape=(output_height, output_width), dtype=np.uint8)
        k = 0
        with torch.no_grad():
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=args.batch_size,
                drop_last=False,
                shuffle=False,
            )
            for input in tqdm(dataloader):

                image = input["img"].cuda()
                image_ids = input["img_id"]

                # Perform sliding window inference
                raw_predictions = sliding_window_inference(
                    model, image, config.num_classes
                )

                # Apply softmax to get class probabilities
                class_probabilities = nn.Softmax(dim=1)(raw_predictions)

                # Get final predictions (class with highest probability for each pixel)
                predictions = class_probabilities.argmax(dim=1)

                for i in range(predictions.shape[0]):
                    mask = predictions[i].cpu().numpy()
                    output_mask = mask

        output_image = os.path.join(args.output_path, img_name)

        cv2.imwrite(output_image, output_mask)

        rasterio_image = rasterio.open(output_image).read(1)

        # Find the corresponding image to get geospatial information
        with rasterio.open(img_path) as src:
            # Get the geospatial information from the corresponding image
            geospatial_info = src.meta
            geospatial_info.update(count=1)  # Keep only one channel

        # Save the stitched image as a geotiff with the updated geospatial information
        with rasterio.open(output_image, "w", **geospatial_info) as dst:
            dst.write(rasterio_image, indexes=1)

        # Save the geotif as shapefile
        os.system(
            f"gdal_polygonize.py -q {output_image} -f 'ESRI Shapefile' {output_image.replace('.tif', '.shp')}"
        )


if __name__ == "__main__":
    main()
