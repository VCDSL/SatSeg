import rasterio
import rasterio.mask
from rasterio.crs import CRS
from rasterio.features import rasterize

import geopandas as gpd
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

import numpy as np


# function of generating binary mask
def generate_mask(raster_path, shape_path):
    """Function that generates a binary mask from a vector file (shp or geojson)

    raster_path = path to the .tif
    shape_path = path to the shapefile or GeoJson
    """

    # Load raster
    with rasterio.open(raster_path, "r") as src:
        raster_img = src.read()
        raster_meta = src.meta

    # Load shapefile as a GeoDataFrame
    train_df = gpd.read_file(shape_path)

    # Verify crs
    if train_df.crs != src.crs:
        print(
            " Raster crs : {}, Vector crs : {}.\n Convert vector and raster to the same CRS.".format(
                src.crs, train_df.crs
            )
        )

    # Function that generates the mask
    def poly_from_utm(polygon, transform):
        poly_pts = []

        poly = unary_union(polygon)
        for i in np.array(poly.exterior.coords):
            poly_pts.append(~transform * tuple(i))

        new_poly = Polygon(poly_pts)
        return new_poly

    poly_shp = []
    im_size = (src.meta["height"], src.meta["width"])
    for _, row in train_df.iterrows():
        if row["geometry"].geom_type == "Polygon":
            poly = poly_from_utm(row["geometry"], src.meta["transform"])
            poly_shp.append(poly)
        else:
            for p in row["geometry"]:
                poly = poly_from_utm(p, src.meta["transform"])
                poly_shp.append(poly)

    mask = rasterize(shapes=poly_shp, out_shape=im_size)

    mask = mask.astype("uint16")

    bin_mask_meta = src.meta.copy()
    bin_mask_meta.update({"count": 1})

    return mask


def tif2np(tif_path: str) -> np.ndarray:
    ds = rasterio.open(tif_path)
    return ds.read()


def get_tif_bounds(tif_path: str, target_crs: int = None):
    ds = rasterio.open(tif_path)
    bounds = ds.bounds

    if target_crs:
        bounds = rasterio.warp.transform_geom(
            ds.crs, CRS.from_epsg(target_crs), box(*bounds)
        )

    return bounds


def pixel_to_crs(pixel_coords: np.ndarray, pixel_bounds: tuple, crs_bounds: tuple):
    """Convert coordinates from pixel domain to CRS domain

    Args:
        pixel_coords (np.ndarray): A numpy array of shape (n, 2) where n is the number of coordinates
        pixel_bounds (tuple): A tuple of 2 integers denoting the maximum number of pixels in each dimension (x, y)
        crs_bounds (tuple): A tuple of 4 floats denoting the bounds of the tif file in CRS coordinates (left, bottom, right, top)
    """
    x, y = pixel_coords[:, 0], pixel_coords[:, 1]

    x = x / pixel_bounds[0] * (crs_bounds[2] - crs_bounds[0]) + crs_bounds[0]
    y = (1 - y / pixel_bounds[1]) * (crs_bounds[3] - crs_bounds[1]) + crs_bounds[1]
