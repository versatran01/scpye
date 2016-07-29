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

tracks = []
init = False
prev_gray = None
init_flow = (38, 0)
proc_cov = (5, 2, 1, 1)

win_size = 31
max_level = 3
flow_cov = (1, 1, 1, 1)
assign_cov = (2, 2)

for i in range(5, 7):
    bw_file = os.path.join(image_dir, bw_name.format(i))
    bgr_file = os.path.join(image_dir, bgr_name.format(i))
    bw = cv2.imread(bw_file, cv2.IMREAD_GRAYSCALE)
    bgr = cv2.imread(bgr_file, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    bc = BinaryCleaner(ksize=3, iters=2, min_area=4)
    bw, region_props = bc.clean(bw)

    disp_bgr = enhance_contrast(bgr)
    disp_bw = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

    ba = BlobAnalyzer()
    fruits = ba.analyze(bgr, region_props)

    draw_bboxes(disp_bgr, fruits, color=(0, 255, 0))
    draw_bboxes(disp_bw, fruits, color=(0, 255, 0))

    if prev_gray is None:
        prev_gray = gray
        for fruit in fruits:
            track = FruitTrack(fruit, init_flow, proc_cov)
            tracks.append(track)
    else:
        # Main loop
        # Prediction
        # get previous points

        # predict
        for track in tracks:
            track.predict()

        pred_bboxes = [t.bbox for t in tracks]
        draw_bboxes(disp_bgr, pred_bboxes, color=(255, 0, 0))
        draw_bboxes(disp_bw, pred_bboxes, color=(255, 0, 0))

        # get predicted points
        prev_points = [t.prev_pos for t in tracks]
        init_points = [t.pos for t in tracks]
        # calculate optical flow
        prev_points, curr_points, status = calc_optical_flow(prev_gray, gray,
                                                             prev_points,
                                                             init_points,
                                                             win_size,
                                                             max_level)
        draw_optical_flows(disp_bgr, prev_points, curr_points, status,
                           color=(255, 0, 255))
        draw_optical_flows(disp_bw, prev_points, curr_points, status,
                           color=(255, 0, 255))
        # update tracks
        updated_tracks, lost_tracks = [], []
        for track, point, stat in izip(tracks, curr_points, status):
            if stat:
                track.correct_flow(point, flow_cov)
                updated_tracks.append(track)
            else:
                lost_tracks.append(track)
        # save image
        prev_gray = gray

        bboxes = [t.bbox for t in updated_tracks]
        ellipses = [t.cov_ellipse for t in updated_tracks]
        draw_bboxes(disp_bgr, bboxes, color=(255, 255, 0))
        draw_bboxes(disp_bw, bboxes, color=(255, 255, 0))

        # assign tracks
        bboxes_update = np.array([t.bbox for t in updated_tracks])
        bboxes_detect = np.array(fruits)
        cost = bboxes_assignment_cost(bboxes_update, bboxes_detect)
        match_inds, lost_inds, new_inds = hungarian_assignment(cost)

        # get matched tracks
        matched_tracks = []
        for match in match_inds:
            i_track, i_fruit = match
            track = tracks[i_track]
            track.correct_assign(fruits[i_fruit], assign_cov)
            matched_tracks.append(track)

        draw_bboxes_matches(disp_bgr, match_inds, bboxes_update,
                            bboxes_detect, color=(0, 0, 255))
        draw_bboxes_matches(disp_bw, match_inds, bboxes_update,
                            bboxes_detect, color=(0, 0, 255))
    imshow(disp_bgr, disp_bw, interp='none', figsize=(12, 16))
