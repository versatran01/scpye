import os
import cv2
import numpy as np
from itertools import izip

from scpye.improc.binary_cleaner import BinaryCleaner
from scpye.improc.blob_analyzer import BlobAnalyzer
from scpye.improc.image_processing import enhance_contrast

from scpye.track.optical_flow import calc_optical_flow
from scpye.track.fruit_track import FruitTrack
from scpye.track.bounding_box import bboxes_assignment_cost
from scpye.track.assignment import hungarian_assignment
from scpye.track.fruit_tracker import FruitTracker
from scpye.utils.data_manager import DataManager
from scpye.utils.drawing import imshow, draw_bboxes
from scpye.utils.drawing import (draw_ellipses, draw_bboxes, draw_points,
                                 draw_optical_flows, draw_bboxes_matches)


# %%
base_dir = '/home/chao/Workspace/dataset/agriculture'
color = 'red'
mode = 'slow_flash'
side = 'north'
bag_ind = 1

# %%
dm = DataManager(base_dir, color=color, mode=mode, side=side)
image_dir = os.path.join(dm.image_dir, "frame" + str(bag_ind))

bw_name = 'bw{0:04d}.png'
bgr_name = 'bgr{0:04d}.png'

bc = BinaryCleaner(ksize=3, iters=2, min_area=4)
ba = BlobAnalyzer()
ft = FruitTracker()

for i in range(5, 8):
    bw_file = os.path.join(image_dir, bw_name.format(i))
    bgr_file = os.path.join(image_dir, bgr_name.format(i))
    bw = cv2.imread(bw_file, cv2.IMREAD_GRAYSCALE)
    bgr = cv2.imread(bgr_file, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    bw, region_props = bc.clean(bw)
    fruits = ba.analyze(bgr, region_props)
    ft.track(bgr, fruits)

#    imshow(disp_bgr, disp_bw, interp='none', figsize=(12, 16))
