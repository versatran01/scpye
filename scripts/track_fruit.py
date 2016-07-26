import os
from itertools import izip

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import watershed

from scpye.processing.binary_cleaner import BinaryCleaner
from scpye.processing.blob_analyzer import BlobAnalyzer
from scpye.processing.contour_analysis import find_contours, moment_centroid
from scpye.processing.image_processing import (u8_from_bw, fill_bw,
                                               local_max_points)
from scpye.utility.data_manager import DataManager
from scpye.utility.visualization import draw_bboxes, draw_points
from scpye.utility.visualization import imshow
from scpye.tracking.bounding_box import extract_bbox

# %%
base_dir = '/home/chao/Workspace/dataset/agriculture'
color = 'red'
mode = 'slow_flash'
side = 'north'
bag_ind = 2

# %%
dm = DataManager(base_dir, color=color, mode=mode, side=side)
image_dir = os.path.join(dm.image_dir, "frame" + str(bag_ind))

i = 100
bw_name = 'bw{0:04d}.png'
bgr_name = 'bgr{0:04d}.png'

bw_file = os.path.join(image_dir, bw_name.format(i))
bgr_file = os.path.join(image_dir, bgr_name.format(i))
bw = cv2.imread(bw_file, cv2.IMREAD_GRAYSCALE)
bgr = cv2.imread(bgr_file, cv2.IMREAD_COLOR)
#imshow(bgr, bw, figsize=(12, 16), interp='none', cmap=plt.cm.viridis)

# %%
bc = BinaryCleaner(ksize=3, iters=2, min_area=5)
bw_filled, props = bc.clean(bw)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
gray[bw_filled == 0] = 0

# %%
disp_bgr = bgr.copy()
disp_bw = cv2.cvtColor(bw_filled, cv2.COLOR_GRAY2BGR)
#imshow(disp_bgr, disp_bw, interp='none', figsize=(12, 16))

# %%
min_dist = 5
# threshold to be a single blob
max_cntr_area = 100
max_aspect = 1.3
min_extent = 0.62
min_solidity = 0.90

blobs = props.blobs
cntrs = props.cntrs

single_blobs, multi_blobs, multi_cntrs = [], [], []
for blob, cntr in izip(blobs, cntrs):
    cntr_area, aspect, extent, solidity = blob['prop']

    if cntr_area < max_cntr_area \
            or (extent > min_extent and aspect < max_aspect) \
            or solidity > min_solidity:
        single_blobs.append(blob)
    else:
        multi_blobs.append(blob)
        multi_cntrs.append(cntr)

single_blobs = np.array(single_blobs)

draw_bboxes(disp_bgr, single_blobs['bbox'])
draw_bboxes(disp_bw, single_blobs['bbox'])
#imshow(disp_bgr, disp_bw, interp='none', figsize=(12, 16))


# %%
def scale_array(data, val=100):
    max_data = np.max(data)
    scale = float(val) / max_data
    return data * scale

# %%
more_single_blobs = []

bboxes = []
single_bboxes = []
for blob, cntr in izip(multi_blobs, multi_cntrs):
    bbox = blob['bbox']

    bgr_bbox = extract_bbox(bgr, bbox)
    gray_bbox = extract_bbox(gray, bbox, copy=True)
    # redraw contour
    bw_cntr = fill_bw(bw, [cntr])
    bw_bbox = extract_bbox(bw_cntr, bbox)
    gray_bbox[bw_bbox == 0] = 0

    gray_blur = ndi.gaussian_filter(gray_bbox, 2)
    distance = ndi.distance_transform_edt(gray_bbox)
    sum_gray_dist = scale_array(gray_blur, val=100) + \
                    scale_array(distance, val=50)
    sum_max = ndi.maximum_filter(sum_gray_dist, size=4, mode='constant')
    local_max = peak_local_max(sum_max, min_distance=5,
                               indices=False, exclude_border=True)
    markers, n_peak = ndi.label(local_max)

    if n_peak > 1:
        labels = watershed(-sum_max, markers, mask=bw_bbox)
        imshow(gray_bbox, distance, sum_max, labels, interp='none',
               figsize=(12, 16), cmap=plt.cm.viridis)
        for i in range(1, n_peak + 1):
            label = u8_from_bw(labels == i)
            local_bbox = np.array(cv2.boundingRect(label))
            local_bbox[:2] += bbox[:2]
            bboxes.append(local_bbox)
    else:
        single_bboxes.append(bbox)

#more_single_blobs = np.array(more_single_blobs)
draw_bboxes(disp_bgr, bboxes, color=(0, 255, 0))
draw_bboxes(disp_bw, bboxes, color=(0, 255, 0))
draw_bboxes(disp_bgr, single_bboxes, color=(255, 255, 0))
draw_bboxes(disp_bw, single_bboxes, color=(255, 255, 0))
imshow(disp_bgr, disp_bw, interp='none', figsize=(12, 16))
