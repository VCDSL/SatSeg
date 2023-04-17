import numpy as np
import math
from tqdm import tqdm
from matplotlib import pyplot as plt
from time import time


def meters_to_latitude(m_dist: float):
    return 0.00001 / 1.11 * m_dist


def degrees_to_radians(angle_deg: float):
    return angle_deg / 180 * math.pi


def radians_to_degrees(angle_deg: float):
    return angle_deg / math.pi * 180


def plot_line(offset: tuple, start: tuple, indices: list, size: tuple):
    while (start[0] < 0 or start[1] < 0) and (start[0] < size[0] and start[1] < size[1]):
        start = (round(start[0] + offset[0]), round(start[1] + offset[1]))
    cur_h = start
        
    while cur_h[0] < size[0] and cur_h[1] < size[1]:
        indices.append(cur_h)
        cur_h = (round(cur_h[0] + offset[0]), round(cur_h[1] + offset[1]))
    return start


def get_grid_score(angle: float, side_len: float, seg_result: np.ndarray, threshold: float = 0.5):
    h, w = seg_result.shape
    
    offset = (int(side_len * math.sin(angle)), int(side_len * math.cos(angle)))
    s_offset_1 = (int(side_len * math.sin(math.pi / 3 + angle)), int(side_len * math.cos(math.pi / 3 + angle)))
    s_offset_2 = (int(side_len * math.sin(math.pi / 3 - angle)), -int(side_len * math.cos(math.pi / 3 - angle)))
    
    cur = (0, 0)
    i = 0
    indices = []
    
    # Lower triangle
    while cur[0] < h and cur[1] < w:
        cur = plot_line(offset, cur, indices, (h, w))
        s_offset = s_offset_1 if i % 2 else s_offset_2
        cur = (round(cur[0] + s_offset[0]), round(cur[1] + s_offset[1]))
        i += 1
    
    
    # Upper triangle
    cur = (0, 0)
    for j in range(i + 3):
        cur = plot_line(offset, cur, indices, (h, w))
        s_offset = s_offset_1 if j % 2 else s_offset_2        
        cur = (round(cur[0] - s_offset[0]), round(cur[1] - s_offset[1]))
        j += 1
    
    indices = np.array(indices)
    seg_result = (seg_result > threshold) * seg_result

    max_score = 0
    best_start = tuple()
    for s_1 in np.arange(0, side_len, 2):
        for s_2 in np.arange(0, side_len, 2):
            indices_new = indices + np.array([s_1, s_2])
            indices_new = indices_new[np.logical_and(indices_new[:, 0] < h, indices_new[:, 1] < w)]
            
            result = np.zeros((h, w))
            result[indices_new[:, 0], indices_new[:, 1]] = seg_result[indices_new[:, 0], indices_new[:, 1]]
            
            score = result.sum()
            if score > max_score:
                max_score = score
                best_start = (s_1, s_2)
    
    # plt.imshow(grid)
    # plt.show()
    # plt.imshow(result)
    # plt.show()
    return max_score, best_start


def make_gaussian(size, fwhm = 10, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 6 + (y - y0) ** 6) / fwhm ** 5)
    

def main():
    mask = make_gaussian(10000, 5000)
    
    angle_range = (0, degrees_to_radians(60))
    angle_res = degrees_to_radians(10)
    side_len_range = (5, 6)
    
    # get_grid_score(0.5235987755982988, 10, (6, 2), mask, 0.8)
    
    max_score = 0
    max_config = tuple()
    for angle in tqdm(np.arange(angle_range[0], angle_range[1], angle_res)):
        for side_len in np.arange(side_len_range[0], side_len_range[1], 1):
            score, start = get_grid_score(angle, side_len, mask, 0.8)
            if score > max_score:
                max_score = score
                max_config = (angle, side_len, start)
    
    print(max_config, max_score)


t = time()
main()
print(f"Search took {time() - t} seconds.")
