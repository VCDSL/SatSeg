import cv2
import numpy as np
from tqdm import tqdm

import fiona
import geopandas as gpd
from pyproj import Transformer
import rasterio
import rasterio.mask
from rasterio.crs import CRS
from rasterio.features import rasterize

from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from satseg.grid_optim import get_optimal_grid


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

    mask = mask.astype("int16")

    bin_mask_meta = src.meta.copy()
    bin_mask_meta.update({"count": 1})

    return mask


def tif2np(tif_path: str) -> np.ndarray:
    ds = rasterio.open(tif_path)
    return ds.read().astype(np.int16)


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


def shapefile_to_latlong(file_path: str):
    c = fiona.open(file_path)
    contours = []

    transformer = Transformer.from_crs(c.crs, 4326, always_xy=True)

    for poly in c:
        coords = poly["geometry"].coordinates
        for coord_set in coords:
            contours.append(
                np.array(
                    list(
                        transformer.itransform(
                            coord_set[0] if len(coord_set) == 1 else coord_set
                        )
                    )
                )
            )

    return contours


def shapefile_to_grid_indices(
    file_path: str, side_len_m: float = 100, meters_per_px: float = 10
):
    c = fiona.open(file_path, "r")
    all_indices = []
    side_len = side_len_m / meters_per_px

    for poly in c:
        coords = poly["geometry"].coordinates
        for coord_set in tqdm(coords):
            contour = np.array(coord_set[0] if len(coord_set) == 1 else coord_set)

            cmin = contour.min(axis=0)
            contour -= cmin
            cmax = int(contour.max() / meters_per_px)
            contour = contour // meters_per_px

            if cmax < side_len:
                continue

            mask = np.zeros((cmax, cmax), dtype="uint8")
            mask = cv2.drawContours(
                mask,
                [contour.reshape((-1, 1, 2)).astype(np.int32)],
                -1,
                (255),
                thickness=cv2.FILLED,
            )

            indices = np.array(get_optimal_grid(mask, side_len=side_len)[0])
            indices = indices * meters_per_px + cmin

            all_indices += indices.tolist()

    transformer = Transformer.from_crs(c.crs, 4326, always_xy=True)
    all_indices = list(transformer.itransform(all_indices))

    return np.array(all_indices)


def points_to_shapefile(points: np.ndarray, file_path: str):
    schema = {"geometry": "Point", "properties": [("ID", "int")]}

    # open a fiona object
    pointShp = fiona.open(
        file_path,
        mode="w",
        driver="ESRI Shapefile",
        schema=schema,
        crs="EPSG:4326",
    )
    # iterate over each row in the dataframe and save record
    for i, (x, y) in enumerate(points):
        rowDict = {
            "geometry": {"type": "Point", "coordinates": (x, y)},
            "properties": {"ID": i},
        }
        pointShp.write(rowDict)
    # close fiona object
    pointShp.close()


def get_cached_grid_indices(file_path: str):
    c = fiona.open(file_path, "r")
    all_indices = []
    all_labels = []

    for poly in c:
        all_indices.append(poly["geometry"].coordinates)
        if "Name" in poly["properties"]:
            all_labels.append(poly["properties"]["Name"])

    return np.array(all_indices), all_labels
