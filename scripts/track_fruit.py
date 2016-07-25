import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scpye.data_manager import DataManager
from scpye.blob_analyzer import BlobAnalyzer
from scpye.visualization import imshow
from scpye.bounding_box import extract_bbox

import scipy.ndimage as ndi
from skimage.feature import peak_local_max

# %%
base_dir = '/home/chao/Workspace/dataset/agriculture'
color = 'red'
mode = 'slow_flash'
side = 'north'
bag_ind = 1

# %%
dm = DataManager(base_dir, color=color, mode=mode, side=side)
image_dir = os.path.join(dm.image_dir, "frame" + str(bag_ind))

i = 133
bw_name = 'bw{0:04d}.png'
bgr_name = 'bgr{0:04d}.png'

bw_file = os.path.join(image_dir, bw_name.format(i))
bgr_file = os.path.join(image_dir, bgr_name.format(i))
bw = cv2.imread(bw_file, cv2.IMREAD_GRAYSCALE)
bgr = cv2.imread(bgr_file, cv2.IMREAD_COLOR)
imshow(bgr, bw, figsize=(12, 16), interp='none', cmap=plt.cm.viridis)

# %%
ba = BlobAnalyzer(ksize=3, iters=2)
blobs, cntrs, gray, filled = ba.analyze(bgr, bw)

bw = np.array(bw > 0, dtype=np.uint8)
filled = np.array(filled > 0, dtype=np.uint8)
imshow(bgr, bw + filled, figsize=(12, 16), interp='none', cmap=plt.cm.viridis)

# %%
Ix = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
Iy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
mag, ang = cv2.cartToPolar(Ix, Iy)
imshow(mag, ang, figsize=(12, 16))

# %%
gray_max = ndi.maximum_filter(gray, size=4, mode='constant')

blob = blobs[40]
bbox = blob['bbox']
bw_bbox = extract_bbox(bw, bbox)
ang_bbox = extract_bbox(ang, bbox)
bgr_bbox = extract_bbox(bgr, bbox)
gray_bbox = extract_bbox(gray, bbox)
gray_bbox_max = ndi.maximum_filter(gray_bbox, size=3, mode='constant')
out = local_max = peak_local_max(gray_bbox_max, min_distance=5, indices=False,
                                 exclude_border=False)
imshow(gray_bbox, bgr_bbox, ang_bbox, out, interp='none', figsize=(12, 18))
