import os
import rasterio
import numpy as np
from shapely.geometry import box
from shapely.strtree import STRtree
import logging
from typing import List, Tuple, Dict
import multiprocessing
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GeoTIFFProcessor:
    def __init__(self, input_folder: str, output_folder: str):
        """
        Initialize the GeoTIFF processor with input and output folders.

        Args:
            input_folder (str): Path to folder containing input GeoTIFF tiles
            output_folder (str): Path to folder for processed outputs
        """
        self.input_folder = input_folder
        self.output_folder = output_folder

        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)

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
        Build a spatial index for efficient neighbor queries.

        Args:
            bounds (List[Tuple[str, box]]): List of (path, bounding box) tuples

        Returns:
            Tuple of spatial index and mapping of geometries to paths
        """
        geometries = [geom for _, geom in bounds]
        spatial_index = STRtree(geometries)
        tile_to_geom = {geom: path for path, geom in bounds}

        return spatial_index, tile_to_geom

    def get_neighbors(
        self,
        spatial_index: STRtree,
        tile_to_geom: Dict[box, str],
        current_tile_path: str,
        min_neighbors: int = 0,  # Optional minimum neighbor requirement
    ) -> List[str]:
        """
        Find neighboring tiles based on spatial overlap.

        Args:
            spatial_index (STRtree): Spatial index of tiles
            tile_to_geom (Dict[box, str]): Mapping of geometries to tile paths
            current_tile_path (str): Path of current tile being processed
            min_neighbors (int): Minimum number of expected neighbors

        Returns:
            List of neighboring tile paths
        """
        with rasterio.open(current_tile_path) as src:
            current_bounds = box(*src.bounds)

        neighbors = spatial_index.query(current_bounds)
        neighbor_paths = [
            tile_to_geom[geom]
            for geom in neighbors
            if tile_to_geom[geom] != current_tile_path
        ]

        # Optional: Raise a warning or handle if too few neighbors
        if len(neighbor_paths) < min_neighbors:
            logger.warning(
                f"Tile {current_tile_path} has fewer than {min_neighbors} neighbors"
            )

        return neighbor_paths

    def tiles_overlap(
        self,
        bounds1: Tuple[float, float, float, float],
        bounds2: Tuple[float, float, float, float],
    ) -> bool:
        """
        Check if two tile bounding boxes overlap.

        Args:
            bounds1 (Tuple): Bounds of first tile (left, bottom, right, top)
            bounds2 (Tuple): Bounds of second tile (left, bottom, right, top)

        Returns:
            bool: Whether tiles overlap
        """
        left1, bottom1, right1, top1 = bounds1
        left2, bottom2, right2, top2 = bounds2

        return not (
            right1 < left2 or left1 > right2 or top1 < bottom2 or bottom1 > top2
        )

    def merge_neighboring_tiles(
        self, center_tile: str, neighbors: List[str]
    ) -> np.ndarray:
        """
        Merge center tile with neighboring tiles.

        Args:
            center_tile (str): Path to the center tile
            neighbors (List[str]): Paths to neighboring tiles

        Returns:
            np.ndarray: Merged array of tiles
        """
        try:
            # Open center tile
            with rasterio.open(center_tile) as center_src:
                center_data = center_src.read()
                center_bounds = center_src.bounds
                center_transform = center_src.transform
                center_crs = center_src.crs

            # Initialize merged array with center tile
            merged_data = center_data.copy()

            # Process each neighbor
            for neighbor_path in neighbors:
                with rasterio.open(neighbor_path) as neighbor_src:
                    neighbor_data = neighbor_src.read()
                    neighbor_bounds = neighbor_src.bounds

                    # Check if tiles actually overlap
                    if self.tiles_overlap(center_bounds, neighbor_bounds):
                        # Align and merge neighbor data
                        merged_data = np.concatenate(
                            [merged_data, neighbor_data], axis=0
                        )

            return merged_data

        except Exception as e:
            logger.error(f"Error merging tiles: {e}")
            return None

    def cutout_and_process_tile(
        self, merged_data: np.ndarray, center_tile_path: str
    ) -> np.ndarray:
        """
        Cut out overlapping areas and process the tile.

        Args:
            merged_data (np.ndarray): Merged tile data
            center_tile_path (str): Path to the center tile

        Returns:
            np.ndarray: Processed tile data
        """
        try:
            with rasterio.open(center_tile_path) as center_tile:
                # Get center tile's shape and bounds
                center_shape = center_tile.shape
                center_bounds = center_tile.bounds

                # Extract center portion from merged data
                # This is a simplified approach and might need refinement
                # depending on your specific tile layout and processing needs
                if merged_data is not None:
                    # Assuming center tile is in the first portion of merged data
                    processed_data = merged_data[
                        :, : center_shape[1], : center_shape[2]
                    ]

                    # Run inference (replace with your actual model)
                    return self.run_inference(processed_data)

                return None
        except Exception as e:
            logger.error(f"Error processing {center_tile_path}: {e}")
            return None

    def run_inference(self, image_data: np.ndarray) -> np.ndarray:
        """
        Placeholder inference method. Replace with your actual model.

        Args:
            image_data (np.ndarray): Input image data

        Returns:
            np.ndarray: Processed image data
        """
        # Example: simple placeholder that returns zeros
        return np.zeros_like(image_data)

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

        neighbors = self.get_neighbors(self.spatial_index, self.tile_to_geom, tile)

        # Merge neighboring tiles
        merged_data = self.merge_neighboring_tiles(tile, neighbors)

        if merged_data is not None:
            try:
                # Process the tile
                result = self.cutout_and_process_tile(merged_data, tile)

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

        # Process tiles
        with multiprocessing.Pool(processes=max_workers) as pool:
            pool.map(self.process_tile, tiles)


def main():
    """
    Main execution function.
    """
    input_folder = "path/to/tiles"
    output_folder = "path/to/output"

    processor = GeoTIFFProcessor(input_folder, output_folder)
    processor.process_all_tiles()


if __name__ == "__main__":
    main()
