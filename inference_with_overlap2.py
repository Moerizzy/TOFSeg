import argparse
from pathlib import Path
import glob
import ttach as tta
import cv2
import numpy as np
import torch
import albumentations as albu
from tools.cfg import py2cfg
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from train_supervision import *
import os
import rasterio
from rasterio.merge import merge

# 1. Load the model folder
# 2. Load a image and find all his neighbors
# 3 Combine the image with his neighbors and discard the outer 2500 pixels
# 4. Run the model on the combined image
# 5. For Inference do slining window but keep always only the inner 70% of the image
# 5. Save the output image, which is the same size as the input image


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser(description="TOFSeg Inference Script")
    parser.add_argument(
        "-i",
        "--image_path",
        required=True,
        help="Path to the folder containing input images.",
    )
    parser.add_argument(
        "-o", "--output_path", required=True, help="Path to save the output masks."
    )
    parser.add_argument(
        "-c", "--config_path", required=True, help="Path to the configuration file."
    )
    parser.add_argument(
        "-st", "--stride", type=int, default=256, help="Stride size for sliding window."
    )
    parser.add_argument(
        "-ps", "--patch_size", type=int, default=1024, help="Patch size for inference."
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=2, help="Batch size for inference."
    )
    return parser.parse_args()


class InferenceDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = sorted(
            [f for f in os.listdir(image_dir) if f.endswith((".tif", ".png", ".jpg"))]
        )
        if not self.image_files:
            raise ValueError(f"No valid images found in directory: {image_dir}")

    def __getitem__(self, index):
        image_name = self.image_files[index]
        image_path = os.path.join(self.image_dir, image_name)
        neighbors = find_neighbors(image_path)
        combined_image = combine_neighbors(neighbors, image_path, (3, 6000, 6000))

        image = np.moveaxis(combined_image, 0, -1).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        image = torch.tensor(image).permute(2, 0, 1).float()
        return {"image": image, "image_name": image_name}

    def __len__(self):
        return len(self.image_files)


def find_neighbors(image_path, radius=500):
    # Get the bounds of the current image
    with rasterio.open(image_path) as src:
        transform = src.transform
        bounds = src.bounds

    neighbors = []

    # Find neighboring GeoTiff-Dateien
    for file in glob.glob(os.path.join(os.path.dirname(image_path), "*.tif")):
        if file != image_path:
            with rasterio.open(file) as src:
                neighbor_bounds = src.bounds

                # Check if the neighbors' bounds intersect with the current image's bounds
                if (
                    neighbor_bounds.left <= bounds.right + radius
                    and neighbor_bounds.right >= bounds.left - radius
                    and neighbor_bounds.bottom <= bounds.top + radius
                    and neighbor_bounds.top >= bounds.bottom - radius
                ):
                    neighbors.append(file)

    return neighbors


def combine_neighbors(neighbors, center_image, output_shape, nodata_value=0):
    """
    Combine neighboring GeoTIFF files into a fixed-size mosaic.

    Parameters:
    - neighbors: List of file paths to GeoTIFF files.
    - output_shape: Tuple (bands, height, width) specifying the fixed output size.
    - nodata_value: Value to fill missing areas (default: 0).

    Returns:
    - combined: A numpy array of the fixed size with combined GeoTIFF data.
    """
    # Create a blank canvas for the output
    combined = np.full(output_shape, nodata_value, dtype=np.float32)

    # Filter valid files
    valid_neighbors = [neighbor for neighbor in neighbors if os.path.exists(neighbor)]

    if not valid_neighbors:
        # If no valid files, place the center image in the center of the blank canvas
        with rasterio.open(center_image) as src:
            center_data = src.read()
        center_h = (output_shape[1] - center_data.shape[1]) // 2
        center_w = (output_shape[2] - center_data.shape[2]) // 2
        combined[
            :,
            center_h : center_h + center_data.shape[1],
            center_w : center_w + center_data.shape[2],
        ] = center_data
        return combined

    # Open valid neighbors
    src_files = [rasterio.open(neighbor) for neighbor in valid_neighbors]

    # Merge tiles
    mosaic, transform = merge(src_files, nodata=nodata_value)

    # Ensure merged data fits into the fixed output size
    min_bands = min(mosaic.shape[0], output_shape[0])
    min_height = min(mosaic.shape[1], output_shape[1])
    min_width = min(mosaic.shape[2], output_shape[2])

    # Copy the merged data into the center of the blank canvas
    combined[:min_bands, :min_height, :min_width] = mosaic[
        :min_bands, :min_height, :min_width
    ]

    # Close all open files
    for src in src_files:
        src.close()

    return combined


def sliding_window_inference_cut(
    model, image, num_classes, patch_size=1024, keep_ratio=0.7
):

    stride = int(patch_size * keep_ratio)
    inner_size = int(patch_size * keep_ratio)
    outer_margin = (patch_size - inner_size) // 2

    _, _, H, W = image.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    image = nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, padded_H, padded_W = image.shape

    prediction = torch.zeros((1, num_classes, padded_H, padded_W), device=image.device)

    for h in range(0, padded_H - patch_size + 1, stride):
        for w in range(0, padded_W - patch_size + 1, stride):
            window = image[:, :, h : h + patch_size, w : w + patch_size]
            with torch.no_grad():
                output = model(window)
            prediction[
                :,
                :,
                h + outer_margin : h + outer_margin + inner_size,
                w + outer_margin : w + outer_margin + inner_size,
            ] = output[
                :,
                :,
                outer_margin : outer_margin + inner_size,
                outer_margin : outer_margin + inner_size,
            ]

    return prediction[:, :, :H, :W]


def sliding_window_inference_overlap(
    model, image, num_classes, patch_size=1024, stride=256
):
    _, _, H, W = image.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    image = nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, padded_H, padded_W = image.shape

    prediction = torch.zeros((1, num_classes, padded_H, padded_W), device=image.device)
    count = torch.zeros((1, 1, padded_H, padded_W), device=image.device)

    for h in range(0, padded_H - patch_size + 1, stride):
        for w in range(0, padded_W - patch_size + 1, stride):
            window = image[:, :, h : h + patch_size, w : w + patch_size]
            with torch.no_grad():
                output = model(window)
            prediction[:, :, h : h + patch_size, w : w + patch_size] += output
            count[:, :, h : h + patch_size, w : w + patch_size] += 1

    prediction /= count
    return prediction[:, :, :H, :W]


def main():
    args = get_args()
    seed_everything(42)

    # Load configuration
    config = py2cfg(args.config_path)
    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name + ".ckpt"),
        config=config,
    )
    model.cuda()
    model.eval()

    # Prepare dataset and dataloader
    dataset = InferenceDataset(image_dir=args.image_path, transform=albu.Normalize())
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    # Inference loop
    for batch in tqdm(dataloader, desc="Processing Images"):
        images = batch["image"].cuda()
        image_names = batch["image_name"]

        predictions = sliding_window_inference_cut(
            model,
            images,
            num_classes=config.num_classes,
            patch_size=args.patch_size,
            keep_ratio=0.7,
        )
        predictions = nn.Softmax(dim=1)(predictions).argmax(dim=1)

        for i, prediction in enumerate(predictions):
            prediction_np = prediction.cpu().numpy().astype(np.uint8)

            center_h = (prediction_np.shape[0] - 5000) // 2
            center_w = (prediction_np.shape[1] - 5000) // 2
            center_prediction = prediction_np[
                center_h : center_h + 5000, center_w : center_w + 5000
            ]

            output_file = os.path.join(args.output_path, image_names[i])

            # Save prediction as GeoTIFF if input is GeoTIFF
            input_path = os.path.join(args.image_path, image_names[i])
            with rasterio.open(input_path) as src:
                meta = src.meta
                meta.update(dtype=rasterio.uint8, count=1)
                with rasterio.open(output_file, "w", **meta) as dst:
                    dst.write(center_prediction, 1)

            # Save the geotif as shapefile
            os.system(
                f"gdal_polygonize.py -q {output_image} -f 'ESRI Shapefile' {output_image.replace('.tif', '.shp')}"
            )


if __name__ == "__main__":
    main()
