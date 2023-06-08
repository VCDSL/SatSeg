import os
import cv2
import numpy as np
from glob import glob


def combine_seg_maps(result_dir: str, save_dir: str):
    metadata = get_result_metadata(result_dir)
    for tif_name in metadata["tifs"]:
        h, w = metadata["tifs"][tif_name]
        size = metadata["size"]
        out_np = np.zeros((h * size, w * size))
        for i in range(h):
            for j in range(w):
                mask = np.load(
                    os.path.join(result_dir, f"{tif_name}_{i * size}_{j * size}.npy")
                )
                out_np[i * size : (i + 1) * size, j * size : (j + 1) * size] = mask

        np.save(os.path.join(save_dir, f"{tif_name}_mask.npy"), out_np.astype(np.uint8))


def get_combined_map_contours(comb_mask_dir: str):
    mask_paths = glob(os.path.join(comb_mask_dir, "*_mask.npy"))
    result_contours = {}
    for path in mask_paths:
        tif_name = os.path.basename(path).split("_")[0]
        mask = np.load(path)
        c_out = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        result_contours[tif_name] = c_out

    return result_contours


def get_result_metadata(result_dir: str) -> dict:
    out_mask_paths = sorted(glob(os.path.join(result_dir, "*_*_*.npy")))
    metadata = {"tifs": {}, "size": np.load(out_mask_paths[0]).shape[0]}

    for path in out_mask_paths:
        tif_name, i, j = os.path.basename(path)[: -len(".npy")].split("_")
        i, j = int(i) // metadata["size"] + 1, int(j) // metadata["size"] + 1

        if tif_name not in metadata["tifs"]:
            metadata["tifs"][tif_name] = (i, j)
        else:
            metadata["tifs"][tif_name] = (
                max(i, metadata["tifs"][tif_name][0]),
                max(j, metadata["tifs"][tif_name][1]),
            )

    return metadata
