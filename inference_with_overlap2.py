import argparse
from pathlib import Path
import glob
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
    parser.add_argument(
        "-m",
        "--margin",
        type=int,
        default=500,
        help="Margin size for neighboring images.",
    )
    parser.add_argument(
        "-kr",
        "--keep_ratio",
        type=float,
        default=0.7,
        help="Ratio of patch to keep in sliding window inference.",
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

        # Get original image dimensions
        with rasterio.open(image_path) as src:
            height = src.height
            width = src.width

        neighbors = find_neighbors(image_path)
        # Use dynamic size based on input image
        combined_image = combine_neighbors(
            neighbors, image_path, (3, height + 1000, width + 1000)
        )

        image = np.moveaxis(combined_image, 0, -1).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        image = torch.tensor(image).permute(2, 0, 1).float()
        return {
            "image": image,
            "image_name": image_name,
            "original_size": (height, width),
        }

    def __len__(self):
        return len(self.image_files)


def find_neighbors(image_path, radius=500):
    with rasterio.open(image_path) as src:
        bounds = src.bounds

    neighbors = []
    for file in glob.glob(os.path.join(os.path.dirname(image_path), "*.tif")):
        if file != image_path:
            with rasterio.open(file) as src:
                neighbor_bounds = src.bounds
                if (
                    neighbor_bounds.left <= bounds.right + radius
                    and neighbor_bounds.right >= bounds.left - radius
                    and neighbor_bounds.bottom <= bounds.top + radius
                    and neighbor_bounds.top >= bounds.bottom - radius
                ):
                    neighbors.append(file)
    return neighbors


def combine_neighbors(neighbors, center_image, output_shape, nodata_value=0):
    combined = np.full(output_shape, nodata_value, dtype=np.float32)
    valid_neighbors = [n for n in neighbors if os.path.exists(n)]

    if not valid_neighbors:
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

    src_files = [rasterio.open(neighbor) for neighbor in valid_neighbors]
    mosaic, transform = merge(src_files, nodata=nodata_value)

    min_bands = min(mosaic.shape[0], output_shape[0])
    min_height = min(mosaic.shape[1], output_shape[1])
    min_width = min(mosaic.shape[2], output_shape[2])

    combined[:min_bands, :min_height, :min_width] = mosaic[
        :min_bands, :min_height, :min_width
    ]

    for src in src_files:
        src.close()

    return combined


def sliding_window_inference(
    model, image, num_classes, patch_size=1024, keep_ratio=0.7
):
    """
    Perform sliding window inference with batch support and proper dimension handling
    """
    stride = int(patch_size * keep_ratio)
    inner_size = int(patch_size * keep_ratio)
    outer_margin = (patch_size - inner_size) // 2

    batch_size, _, H, W = image.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    image = nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, padded_H, padded_W = image.shape

    # Initialize tensors with correct dimensions
    prediction = torch.zeros(
        (batch_size, num_classes, padded_H, padded_W), device=image.device
    )
    count = torch.zeros(
        (batch_size, num_classes, padded_H, padded_W), device=image.device
    )

    for h in range(0, padded_H - patch_size + 1, stride):
        for w in range(0, padded_W - patch_size + 1, stride):
            window = image[:, :, h : h + patch_size, w : w + patch_size]
            with torch.no_grad():
                output = model(window)

            # Update predictions and counts
            prediction[
                :,
                :,
                h + outer_margin : h + outer_margin + inner_size,
                w + outer_margin : w + outer_margin + inner_size,
            ] += output[
                :,
                :,
                outer_margin : outer_margin + inner_size,
                outer_margin : outer_margin + inner_size,
            ]
            count[
                :,
                :,
                h + outer_margin : h + outer_margin + inner_size,
                w + outer_margin : w + outer_margin + inner_size,
            ] += 1

    # Average the predictions where windows overlap
    valid_mask = count > 0
    prediction = torch.where(valid_mask, prediction / count, prediction)

    return prediction[:, :, :H, :W]


def main():
    args = get_args()
    seed_everything(42)

    config = py2cfg(args.config_path)
    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name + ".ckpt"),
        config=config,
    )
    model.cuda()
    model.eval()

    dataset = InferenceDataset(image_dir=args.image_path, transform=albu.Normalize())
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    os.makedirs(args.output_path, exist_ok=True)

    for batch in tqdm(dataloader, desc="Processing Images"):
        images = batch["image"].cuda()
        image_names = batch["image_name"]
        original_sizes = batch["original_size"]

        predictions = sliding_window_inference(
            model,
            images,
            num_classes=config.num_classes,
            patch_size=args.patch_size,
            keep_ratio=args.keep_ratio,
        )
        predictions = nn.Softmax(dim=1)(predictions).argmax(dim=1)

        for i, prediction in enumerate(predictions):
            prediction_np = prediction.cpu().numpy().astype(np.uint8)
            orig_h, orig_w = original_sizes[i]

            # Extract the center portion matching the original image size
            center_h = (prediction_np.shape[0] - orig_h) // 2
            center_w = (prediction_np.shape[1] - orig_w) // 2
            center_prediction = prediction_np[
                center_h : center_h + orig_h, center_w : center_w + orig_w
            ]

            output_file = os.path.join(args.output_path, image_names[i])

            # Save prediction as GeoTIFF
            input_path = os.path.join(args.image_path, image_names[i])
            with rasterio.open(input_path) as src:
                meta = src.meta
                meta.update(dtype=rasterio.uint8, count=1)
                with rasterio.open(output_file, "w", **meta) as dst:
                    dst.write(center_prediction, 1)

            # Convert to shapefile
            os.system(
                f"gdal_polygonize.py -q {output_file} -f 'ESRI Shapefile' "
                f"{output_file.replace('.tif', '.shp')}"
            )


if __name__ == "__main__":
    main()
