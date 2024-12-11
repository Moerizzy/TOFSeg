import os
import rasterio
from rasterio.merge import merge
from rasterio.windows import Window
import numpy as np
from shapely.geometry import box
from shapely.strtree import STRtree
import logging
from typing import List, Tuple, Dict
import multiprocessing
import tempfile
from osgeo import gdal

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

        # Disable GDAL/PROJ warnings
        gdal.UseExceptions()
        gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "TRUE")
        gdal.SetConfigOption("GDAL_CACHEMAX", "20%")

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
    ) -> List[str]:
        """
        Find neighboring tiles based on spatial overlap.

        Args:
            spatial_index (STRtree): Spatial index of tiles
            tile_to_geom (Dict[box, str]): Mapping of geometries to tile paths
            current_tile_path (str): Path of current tile being processed

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

        return neighbor_paths

    def create_vrt(self, center_tile: str, neighbors: List[str]) -> str:
        """
        Create a Virtual Raster (VRT) from center tile and neighbors.

        Args:
            center_tile (str): Path to the center tile
            neighbors (List[str]): Paths to neighboring tiles

        Returns:
            str: Path to created VRT file
        """
        with tempfile.NamedTemporaryFile(suffix=".vrt", delete=False) as temp_vrt:
            vrt_path = temp_vrt.name

        try:
            vrt_options = gdal.BuildVRTOptions(resolution="highest", addAlpha=True)
            gdal.BuildVRT(vrt_path, [center_tile] + neighbors, options=vrt_options)
            return vrt_path
        except Exception as e:
            logger.error(f"Error creating VRT: {e}")
            return None

    def cutout_and_process_tile(
        self, vrt_path: str, center_tile_path: str
    ) -> np.ndarray:
        """
        Cut out overlapping areas and process the tile.

        Args:
            vrt_path (str): Path to the VRT file
            center_tile_path (str): Path to the center tile

        Returns:
            np.ndarray: Processed tile data
        """
        try:
            with rasterio.open(vrt_path) as vrt, rasterio.open(
                center_tile_path
            ) as center_tile:
                # Get window for the center tile
                center_window = vrt.window(*center_tile.bounds)

                # Expand the window to include neighbors
                expanded_window = Window(
                    col_off=max(0, center_window.col_off - center_window.width // 2),
                    row_off=max(0, center_window.row_off - center_window.height // 2),
                    width=center_window.width * 2,
                    height=center_window.height * 2,
                )

                # Read expanded area
                expanded_area = vrt.read(window=expanded_window)

                # Run inference (replace with your actual model)
                return self.run_inference(expanded_area)
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

        # Create VRT for the tile and its neighbors
        vrt_path = self.create_vrt(tile, neighbors)

        if vrt_path:
            try:
                # Process the tile
                result = self.cutout_and_process_tile(vrt_path, tile)

                # Save the result
                self.save_result(result, tile)
            finally:
                # Clean up temporary VRT
                if os.path.exists(vrt_path):
                    os.unlink(vrt_path)

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
