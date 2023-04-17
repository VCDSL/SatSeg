import os
import cv2
import numpy as np
from glob import glob


def combine_seg_maps(result_dir: str, save_dir: str):
    metadata = get_result_metadata(result_dir)
    for tif_name in metadata['tifs']:
        h, w = metadata['tifs'][tif_name]
        size = metadata['size']
        out_np = np.zeros((h * size, w * size))
        for i in range(h):
            for j in range(w):
                mask = np.load(os.path.join(result_dir, f'{tif_name}_{i}_{j}.npy'))
                out_np[i * size: (i + 1) * size, j * size: (j + 1) * size] = mask
    
        np.save(os.path.join(save_dir, f'{tif_name}_mask.npy'))


def get_result_contours(result: np.ndarray):
    contours, _ = cv2.findContours(result)
    return contours
    

def get_result_metadata(result_dir: str) -> dict:
    out_mask_paths = sorted(glob(os.path.join(result_dir, '*_*_*.npy')))
    metadata = {'tifs': {}, 'size': np.load(out_mask_paths[0]).shape[0]}
    
    for path in out_mask_paths:
        tif_name, i, j = os.path.basename(path)[:-len('.npy')].split('_')
        i, j = int(i) + 1, int(j) + 1
        
        if tif_name not in metadata['tifs']:
            metadata['tifs'][tif_name] = (i, j)
        else:
            metadata['tifs'][tif_name] = (max(i, metadata['tifs'][tif_name][0]), max(j, metadata['tifs'][tif_name][1]))
    
    return metadata
    
    