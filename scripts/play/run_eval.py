# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 16:15:28 2016

@author: chao
"""

# %%
from __future__ import division

import sys

# HACK
from scpye.region_props import gray_from_bw

sys.path.append('..')

import os
from sklearn.externals import joblib
from scpye.training import *
from scpye.blob_analyzer import *
from scpye.visualization import *
from scpye.bounding_box import extract_bbox

# %%
base_dir = '/home/chao/Dropbox'
color = 'red'
mode = 'slow_flash'
train_inds = range(0, 12, 3)
test_inds = range(1, 12, 3)
save = True
load = False

drd = DataReader(base_dir=base_dir, color=color, mode=mode)
img_ppl_pkl = os.path.join(drd.model_dir, 'img_ppl.pkl')
img_clf_pkl = os.path.join(drd.model_dir, 'img_clf.pkl')

# %%
img_ppl = joblib.load(img_ppl_pkl)
img_clf = joblib.load(img_clf_pkl)

# %%
I, L = drd.load_image_label(2)
img_ppl.transform(I, L)

# Get transformed label
lbl = img_ppl.named_steps['remove_dark'].label
pos = lbl[:, :, 1]
pos = gray_from_bw(pos)

# %%
X = img_ppl.transform(I)
y = img_clf.predict(X)

hsv = img_ppl.named_features['hsv'].image
bgr = img_ppl.named_steps['remove_dark'].image
bw = img_ppl.named_steps['remove_dark'].mask.copy()
bw[bw] = y
bw = gray_from_bw(bw)
# Do a cleaning first
bw = clean_bw(bw)

# Get only reasonable blobs and redraw bw
blbs, cnts = region_props(bw)
bw = fill_bw(bw, cnts)

disp_bw = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
disp_color = np.array(bgr, copy=True)

imshow2(disp_color, disp_bw)

# %%
from skimage.feature import peak_local_max

blob = blbs[34]
bbox = blob['bbox']

bgr_bbox = extract_bbox(bgr, bbox, copy=True)
v_bbox = extract_bbox(hsv[:, :, -1], bbox)
bw_bbox = extract_bbox(bw, bbox)
v_bbox[bw_bbox == 0] = 0
imshow2(v_bbox, bw_bbox)

# This returns [row, column], need to flip it
rc = peak_local_max(v_bbox, min_distance=5)
uv = np.fliplr(rc)
draw_point(bgr_bbox, uv)

imshow(bgr_bbox)

# %%
# skiamge watershed
from scipy import ndimage
from skimage.morphology import watershed

dist = ndimage.distance_transform_edt(bw_bbox)
local_max = peak_local_max(dist, indices=False, min_distance=5)
imshow(local_max)

markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
labels = watershed(-dist, markers, mask=bw_bbox)
imshow(labels)

# %%
# opencv watershed
# http://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html#gsc.tab=0
# Opencv watershed is annoying since it requires 3 channel which is unnecessary