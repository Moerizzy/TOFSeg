import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import mapping
import os

from tof.utils import (
    get_file_list,
    argparse_TOF,
)

if __name__ == "__main__":

    # ignore warnings

    args = argparse_TOF()
    state = args.state
    epsg = args.epsg

    list_path_RGBI = get_file_list(f"Sites/{state}/TOP")
    path_Shape = f"Sites/{state}/SHP_noALKIS/{state}_result_merged_reclassified.shp"

    gdf = gpd.read_file(path_Shape)

    # Check if mask folder exists
    if not os.path.exists(f"Sites/{state}/Masks"):
        os.makedirs(f"Sites/{state}/Masks")

    for i, path_RGBI in enumerate(list_path_RGBI):

        name = path_RGBI.split("/")[-1].replace("TOP", "mask")
        mask_path = f"Sites/{state}/Masks/{name}"

        with rasterio.open(path_RGBI[:-4] + ".tif") as ref_raster:
            with rasterio.open(
                mask_path,
                "w",
                driver="GTiff",
                height=ref_raster.height,
                width=ref_raster.width,
                count=1,
                dtype=rasterio.uint16,
                crs=ref_raster.crs,
                transform=ref_raster.transform,
            ) as mask_raster:
                burned = rasterize(
                    [
                        (geometry, value)
                        for geometry, value in zip(gdf.geometry, gdf["classvalue"])
                    ],
                    out_shape=ref_raster.shape,
                    all_touched=False,
                    transform=ref_raster.transform,
                    fill=0,
                    dtype=rasterio.uint16,
                    default_value=1,
                )

                mask_raster.write(burned, 1)

        print(f"{i+1}/{len(list_path_RGBI)}: {name} created at {mask_path}")
