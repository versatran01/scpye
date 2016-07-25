import os
from itertools import izip

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from skimage.feature import peak_local_max

from scpye.processing.binary_cleaner import BinaryCleaner
from scpye.processing.blob_analyzer import BlobAnalyzer
from scpye.utility.data_manager import DataManager
from scpye.utility.visualization import draw_bboxes
from scpye.utility.visualization import imshow
from scpye.tracking.bounding_box import extract_bbox


# %%
base_dir = '/home/chao/Workspace/dataset/agriculture'
color = 'red'
mode = 'slow_flash'
side = 'north'
bag_ind = 1

# %%
dm = DataManager(base_dir, color=color, mode=mode, side=side)
image_dir = os.path.join(dm.image_dir, "frame" + str(bag_ind))

i = 120
bw_name = 'bw{0:04d}.png'
bgr_name = 'bgr{0:04d}.png'

bw_file = os.path.join(image_dir, bw_name.format(i))
bgr_file = os.path.join(image_dir, bgr_name.format(i))
bw = cv2.imread(bw_file, cv2.IMREAD_GRAYSCALE)
bgr = cv2.imread(bgr_file, cv2.IMREAD_COLOR)
imshow(bgr, bw, figsize=(12, 16), interp='none', cmap=plt.cm.viridis)

# %%
bc = BinaryCleaner(ksize=3, iters=2, min_area=5)
bw_filled, props = bc.clean(bw)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
gray[bw == 0] = 0

# %%

disp_bgr = bgr.copy()
disp_bw = cv2.cvtColor(bw_filled, cv2.COLOR_GRAY2BGR)
imshow(disp_bgr, disp_bw, interp='none', figsize=(12, 16))

min_dist = 5
# threshold to be a single blob
max_cntr_area = 100
max_aspect = 1.3
min_extent = 0.62
min_solidity = 0.90

blobs = props.blobs
cntrs = props.cntrs

single_blobs, multi_blobs = [], []
for blob, cntr in izip(blobs, cntrs):
    cntr_area, aspect, extent, solidity = blob['prop']

    if cntr_area < max_cntr_area \
            or (extent > min_extent and aspect < max_aspect) \
            or solidity > min_solidity:
        single_blobs.append(blob)
    else:
        multi_blobs.append(blob)

single_blobs = np.array(single_blobs)

draw_bboxes(disp_bgr, single_blobs['bbox'])
draw_bboxes(disp_bw, single_blobs['bbox'])
imshow(disp_bgr, disp_bw, interp='none', figsize=(12, 16))

# %%
false_multi_blobs = []
for blob in multi_blobs:
    bbox = blob['bbox']
    bgr_bbox = extract_bbox(bgr, bbox)
    gray_bbox = extract_bbox(gray, bbox)
    gray_bbox_max = ndi.maximum_filter(gray_bbox, size=min_dist, mode='constant')
    local_max = peak_local_max(gray_bbox_max, min_distance=min_dist,
                               indices=False, exclude_border=True)
    imshow(bgr_bbox, gray_bbox_max, local_max, figsize=(10, 16), interp='none',
           cmap=plt.cm.viridis)
