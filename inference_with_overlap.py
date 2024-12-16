import os
import rasterio
import numpy as np
from shapely.geometry import box
from shapely.strtree import STRtree
import logging
from typing import List, Tuple, Dict
import multiprocessing
import argparse
import time
import torch  # Assuming a PyTorch model, adjust as needed

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GeoTIFFProcessor:
    def __init__(self, input_folder: str, output_folder: str, model_path: str):
        """
        Initialize the GeoTIFF processor with input, output folders, and model.

        Args:
            input_folder (str): Path to folder containing input GeoTIFF tiles
            output_folder (str): Path to folder for processed outputs
            model_path (str): Path to the trained model file
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.model_path = model_path

        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Load the model
        self.model = self.load_model()

    def load_model(self):
        """
        Load the machine learning model.

        Returns:
            Loaded model for inference
        """
        try:
            # Example for PyTorch model loading
            model = torch.load(self.model_path)
            model.eval()  # Set to evaluation mode
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def get_tiles_and_bounds(self) -> Tuple[List[str], List[Tuple[str, box]]]:
        """
        Gather GeoTIFF files and their bounding boxes.

        Returns:
            Tuple of tiles list and bounds list
        """
        tiles = []
        bounds = []

        for file in os.listdir(self.input_folder):
            if file.endswith(".tif"):
                file_path = os.path.join(self.input_folder, file)
                try:
                    with rasterio.open(file_path) as src:
                        tiles.append(file_path)
                        bounds.append((file_path, box(*src.bounds)))
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")

        return tiles, bounds

    def build_spatial_index(
        self, bounds: List[Tuple[str, box]]
    ) -> Tuple[STRtree, Dict[box, str]]:
        """
        Build a spatial index for identifying adjacent tiles.

        Args:
            bounds (List[Tuple[str, box]]): List of (path, bounding box) tuples

        Returns:
            Tuple of spatial index and mapping of geometries to paths
        """
        geometries = [geom for _, geom in bounds]
        spatial_index = STRtree(geometries)
        tile_to_geom = {geom: path for path, geom in bounds}

        return spatial_index, tile_to_geom

    def tiles_are_adjacent(
        self,
        bounds1: Tuple[float, float, float, float],
        bounds2: Tuple[float, float, float, float],
    ) -> bool:
        """
        Check if two tiles are directly adjacent (sharing an edge, no overlap).

        Args:
            bounds1 (Tuple): Bounds of first tile (left, bottom, right, top)
            bounds2 (Tuple): Bounds of second tile (left, bottom, right, top)

        Returns:
            bool: Whether tiles are directly adjacent
        """
        left1, bottom1, right1, top1 = bounds1
        left2, bottom2, right2, top2 = bounds2

        # Check horizontal adjacency (left-right or right-left)
        horizontal_adjacent = (
            abs(right1 - left2) < 1e-6 or abs(right2 - left1) < 1e-6
        ) and not (top1 < bottom2 or bottom1 > top2)

        # Check vertical adjacency (top-bottom or bottom-top)
        vertical_adjacent = (
            abs(top1 - bottom2) < 1e-6 or abs(top2 - bottom1) < 1e-6
        ) and not (right1 < left2 or left1 > right2)

        return horizontal_adjacent or vertical_adjacent

    def get_adjacent_tiles(
        self,
        spatial_index: STRtree,
        tile_to_geom: Dict[box, str],
        current_tile_path: str,
    ) -> List[str]:
        """
        Find directly adjacent tiles.

        Args:
            spatial_index (STRtree): Spatial index of tiles
            tile_to_geom (Dict[box, str]): Mapping of geometries to tile paths
            current_tile_path (str): Path of current tile being processed

        Returns:
            List of adjacent tile paths
        """
        with rasterio.open(current_tile_path) as src:
            current_bounds = box(*src.bounds)

        adjacent_tiles = []
        for geom in spatial_index.query(current_bounds):
            candidate_tile_path = tile_to_geom[geom]

            if candidate_tile_path != current_tile_path:
                with rasterio.open(candidate_tile_path) as other_src:
                    other_bounds = box(*other_src.bounds)

                    if self.tiles_are_adjacent(
                        current_bounds.bounds, other_bounds.bounds
                    ):
                        adjacent_tiles.append(candidate_tile_path)

        return adjacent_tiles

    def merge_adjacent_tiles(
        self, center_tile: str, adjacent_tiles: List[str]
    ) -> np.ndarray:
        """
        Merge center tile with adjacent tiles.

        Args:
            center_tile (str): Path to the center tile
            adjacent_tiles (List[str]): Paths to adjacent tiles

        Returns:
            np.ndarray: Merged tile data
        """
        try:
            # List of all tiles to merge (including center tile)
            all_tiles = [center_tile] + adjacent_tiles

            # Open the first tile to get reference metadata
            with rasterio.open(center_tile) as first_src:
                # Get metadata from the first tile
                profile = first_src.profile.copy()
                dtype = first_src.dtypes[0]

                # Determine the merged raster dimensions
                # This assumes tiles are aligned and have the same resolution
                width = first_src.width * (1 if len(adjacent_tiles) == 0 else 3)
                height = first_src.height * (1 if len(adjacent_tiles) == 0 else 3)

                # Create a new transform that covers the entire merged area
                new_transform = first_src.transform * rasterio.Affine.translation(
                    -first_src.width, -first_src.height
                )

                # Update the profile for the merged raster
                profile.update(
                    {"width": width, "height": height, "transform": new_transform}
                )

            # Create an output array for the merged tiles
            merged_data = np.zeros((first_src.count, height, width), dtype=dtype)

            # Tile positions (assuming 3x3 grid with center tile in the middle)
            tile_positions = [
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
                (1, 2),
                (2, 0),
                (2, 1),
                (2, 2),
            ]

            # Merge tiles
            for idx, tile_path in enumerate(all_tiles):
                with rasterio.open(tile_path) as src:
                    tile_data = src.read()

                    # Calculate position in the merged array
                    row, col = tile_positions[idx]
                    tile_height, tile_width = tile_data.shape[1], tile_data.shape[2]

                    # Place the tile in the correct position
                    merged_data[
                        :,
                        row * tile_height : (row + 1) * tile_height,
                        col * tile_width : (col + 1) * tile_width,
                    ] = tile_data

            return merged_data

        except Exception as e:
            logger.error(f"Error merging tiles: {e}")
            return None

    def load_model(self):
        """
        Load the machine learning model.

        Returns:
            Loaded model for inference
        """
        try:
            # Example for PyTorch model loading
            model = torch.load(self.model_path)
            model.cuda()  # Move to GPU
            model.eval()  # Set to evaluation mode
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def sliding_window_inference(
        self, image: torch.Tensor, patch_size: int = 1024, stride: int = 256
    ) -> torch.Tensor:
        """
        Perform sliding window inference on a tile.

        Args:
            image (torch.Tensor): Input image tensor
            patch_size (int): Size of inference patches
            stride (int): Sliding window stride

        Returns:
            torch.Tensor: Predicted segmentation map
        """
        # Ensure the model is in evaluation mode
        self.model.eval()

        # Handle single image case and ensure 4D tensor
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        # Get image dimensions
        _, _, H, W = image.shape

        # Determine number of classes from model (you might need to adjust this)
        num_classes = (
            self.model.num_classes if hasattr(self.model, "num_classes") else 2
        )

        # Pad image if needed
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size

        image = nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="reflect")

        # Get padded dimensions
        _, _, padded_H, padded_W = image.shape

        # Initialize prediction and count tensors
        prediction = torch.zeros(
            (1, num_classes, padded_H, padded_W), device=image.device
        )
        count = torch.zeros((1, 1, padded_H, padded_W), device=image.device)

        # Sliding window inference
        for h in range(0, padded_H - patch_size + 1, stride):
            for w in range(0, padded_W - patch_size + 1, stride):
                # Extract window
                window = image[:, :, h : h + patch_size, w : w + patch_size]

                # Inference on window
                with torch.no_grad():
                    output = self.model(window)

                # Accumulate predictions
                prediction[:, :, h : h + patch_size, w : w + patch_size] += output
                count[:, :, h : h + patch_size, w : w + patch_size] += 1

        # Normalize predictions
        prediction /= count

        # Crop back to original image size
        return prediction[:, :, :H, :W]

    def run_inference(self, image_data: np.ndarray) -> np.ndarray:
        """
        Run inference using the loaded model with sliding window approach.

        Args:
            image_data (np.ndarray): Input image data

        Returns:
            np.ndarray: Processed image data
        """
        try:
            # Convert numpy array to PyTorch tensor
            # Ensure correct shape and normalization
            tensor = torch.from_numpy(image_data).float().cuda()

            # Normalize if needed (adjust based on your model's requirements)
            tensor = tensor / 255.0  # Normalize to [0, 1]

            # Run sliding window inference
            result = self.sliding_window_inference(tensor)

            # Convert to class predictions
            predictions = nn.Softmax(dim=1)(result).argmax(dim=1)

            # Convert back to numpy
            return predictions.cpu().numpy().astype(np.uint8)

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return None

    def save_result(self, result: np.ndarray, center_tile_path: str):
        """
        Save processing result.

        Args:
            result (np.ndarray): Processed tile data
            center_tile_path (str): Path to the original center tile
        """
        if result is None:
            logger.warning(f"No result to save for {center_tile_path}")
            return

        output_path = os.path.join(
            self.output_folder, os.path.basename(center_tile_path)
        )

        try:
            # Open the original tile to copy metadata
            with rasterio.open(center_tile_path) as src:
                profile = src.profile.copy()

            # Update profile for the output
            profile.update(
                dtype=result.dtype,
                count=result.shape[0],
                height=result.shape[1],
                width=result.shape[2],
            )

            # Write the result
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(result)

            logger.info(f"Saved result to {output_path}")
        except Exception as e:
            logger.error(f"Error saving result for {center_tile_path}: {e}")

    def process_tile(self, tile: str):
        """
        Process a single tile.

        Args:
            tile (str): Path to the tile to process
        """
        logger.info(f"Processing tile: {tile}")

        # Get adjacent tiles
        adjacent_tiles = self.get_adjacent_tiles(
            self.spatial_index, self.tile_to_geom, tile
        )

        # Prepare tile data
        prepared_data = self.merge_adjacent_tiles(tile, adjacent_tiles)

        if prepared_data is not None:
            try:
                # Process the tile
                result = self.run_inference(prepared_data)

                # Save the result
                self.save_result(result, tile)
            except Exception as e:
                logger.error(f"Error processing tile {tile}: {e}")

    def process_all_tiles(self, max_workers: int = None):
        """
        Process all tiles, optionally in parallel.

        Args:
            max_workers (int, optional): Number of parallel workers.
                                         Defaults to number of CPU cores.
        """
        # Get tiles and build spatial index
        tiles, bounds = self.get_tiles_and_bounds()
        self.spatial_index, self.tile_to_geom = self.build_spatial_index(bounds)

        # Determine number of workers
        if max_workers is None:
            max_workers = os.cpu_count()

        def timed_process_tile(tile):
            start_time = time.time()
            self.process_tile(tile)
            end_time = time.time()
            logger.info(
                f"Processing time for {tile}: {end_time - start_time:.2f} seconds"
            )

        # Process tiles
        with multiprocessing.Pool(processes=max_workers) as pool:
            pool.map(timed_process_tile, tiles)


def main():
    """
    Main execution function with argument parsing.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Process GeoTIFF tiles with adjacent tile awareness"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to input folder containing GeoTIFF tiles",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to output folder for processed tiles",
    )
    parser.add_argument(
        "-m", "--model", required=True, help="Path to the trained model file"
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU core count)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Create processor and process tiles
    processor = GeoTIFFProcessor(
        input_folder=args.input, output_folder=args.output, model_path=args.model
    )
    processor.process_all_tiles(max_workers=args.workers)


if __name__ == "__main__":
    main()
