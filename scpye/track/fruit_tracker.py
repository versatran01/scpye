from __future__ import (print_function, division, absolute_import)

import cv2
import numpy as np
from itertools import izip

from scpye.track.assignment import hungarian_assignment
from scpye.track.bounding_box import bboxes_assignment_cost
from scpye.track.fruit_track import FruitTrack
from scpye.track.optical_flow import calc_optical_flow
from scpye.improc.image_processing import enhance_contrast

from scpye.utils.drawing import (Colors, draw_bboxes, draw_optical_flows,
                                 draw_bboxes_matches)


class FruitTracker(object):
    def __init__(self, min_age=3, win_size=31, max_level=3, init_flow=(40, 0),
                 state_cov=(5, 5, 5, 5), proc_cov=(10, 4, 5, 2),
                 flow_cov=(3, 3),
                 bbox_cov=(1, 1)):
        """
        :param min_age: minimum age of a tracking to be considered for counting
        """
        # Tracking
        self.tracks = []
        self.min_age = min_age
        self.total_counts = 0

        self.prev_gray = None
        # Optical flow parameters
        self.win_size = win_size
        self.max_level = max_level

        # Kalman filter parameters
        self.state_cov = np.array(state_cov)
        self.init_flow = np.array(init_flow)
        self.proc_cov = np.array(proc_cov)
        self.flow_cov = np.array(flow_cov)
        self.bbox_cov = np.array(bbox_cov)

        # Visualization
        self.vis = True
        self.disp_bgr = None
        self.disp_bw = None

    @property
    def initialized(self):
        return self.prev_gray is not None

    def add_new_tracks(self, tracks, fruits):
        """
        Add new fruits to tracks
        :param tracks: a list of tracks
        :param fruits: a list of fruits
        """
        for fruit in fruits:
            track = FruitTrack(fruit, self.init_flow, self.state_cov,
                               self.proc_cov)
            tracks.append(track)

    def track(self, image, fruits, bw):
        """
        Main tracking step
        :param image: greyscale image
        :param fruits: new fruits
        :param bw: binary image
        """
        # Convert to greyscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        self.disp_bgr = enhance_contrast(image)
        self.disp_bw = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

        # VISUALIZATION: detect
        if self.vis:
            draw_bboxes(self.disp_bgr, fruits, color=Colors.white)
            draw_bboxes(self.disp_bw, fruits, color=Colors.white)

        # Initialization
        if not self.initialized:
            self.prev_gray = gray
            self.add_new_tracks(self.tracks, fruits)
            print('Initialization done.')
            return

        self.predict_tracks()

        # VISUALIZATION: after prediction
        if self.vis:
            predict_bboxes = [t.bbox for t in self.tracks]
            draw_bboxes(self.disp_bgr, predict_bboxes, color=Colors.red)
            draw_bboxes(self.disp_bw, predict_bboxes, color=Colors.red)

        updated_tracks, lost_tracks = self.update_tracks(gray)

        # VISUALIZATION: after optical flow update
        if self.vis:
            updated_bboxes = [t.bbox for t in updated_tracks]
            draw_bboxes(self.disp_bgr, updated_bboxes, color=Colors.yellow)
            draw_bboxes(self.disp_bw, updated_bboxes, color=Colors.yellow)

        matched_tracks, unmatched_tks = self.match_tracks(updated_tracks,
                                                          fruits)

        # VISUALIZATION: after hungarian assignment update
        if self.vis:
            matched_bboxes = [t.bbox for t in matched_tracks]
            draw_bboxes(self.disp_bgr, matched_bboxes, color=Colors.green)
            draw_bboxes(self.disp_bw, matched_bboxes, color=Colors.green)

        # Assemble all lost tracks and update tracks
        self.tracks = matched_tracks
        lost_tracks.extend(unmatched_tks)

        # Count fruits in lost tracks
        # self.count_in_tracks(lost_tks)

    def predict_tracks(self):
        """
        Predict tracks in Kalman filter
        """
        for track in self.tracks:
            track.predict()

    def update_tracks(self, gray):
        """
        Update tracks' position in Kalman filter via KLT
        :param gray: greyscale image
        :return: updated tracks, lost_tracks
        """
        # Get points in previous image and points in current image
        prev_pts = [t.prev_pos for t in self.tracks]
        init_pts = [t.pos for t in self.tracks]

        curr_pts, status = calc_optical_flow(self.prev_gray, gray,
                                             prev_pts, init_pts,
                                             self.win_size,
                                             self.max_level)

        # VISUALIZATION: optical flow
        draw_optical_flows(self.disp_bgr, prev_pts, curr_pts, status,
                           color=Colors.magenta)
        draw_optical_flows(self.disp_bw, prev_pts, curr_pts, status,
                           color=Colors.magenta)

        # Remove lost tracks
        updated_tracks, lost_tracks = [], []
        for track, point, stat in izip(self.tracks, curr_pts, status):
            if stat:
                track.correct_flow(point, self.flow_cov)
                updated_tracks.append(track)
            else:
                lost_tracks.append(track)

        self.prev_gray = gray

        return updated_tracks, lost_tracks

    def match_tracks(self, tracks, fruits):
        """
        Match tracks to new detection
        :param tracks:
        :param fruits:
        :return: matched_tracks, unmatched_tracks
        """
        bboxes_update = np.array([t.bbox for t in tracks])
        bboxes_detect = np.array(fruits)

        cost = bboxes_assignment_cost(bboxes_update, bboxes_detect)
        match_inds, lost_inds, new_inds = hungarian_assignment(cost)

        # VISUALIZATION: hungarian assignment
        # draw_bboxes_matches(self.disp_bgr, match_inds, bboxes_update,
        #                     bboxes_detect, color=Colors.cyan)

        # get matched tracks
        matched_tracks = []
        for match in match_inds:
            i_track, i_fruit = match
            track = tracks[i_track]
            track.correct_bbox(fruits[i_fruit], self.bbox_cov)
            matched_tracks.append(track)

        # add new tracks
        new_fruits = fruits[new_inds]
        self.add_new_tracks(matched_tracks, new_fruits)

        # get unmatched tracks
        unmatched_tracks = [tracks[ind] for ind in lost_inds]

        return matched_tracks, unmatched_tracks

    def count_in_tracks(self, tracks):
        """
        Count how many fruits there are in tracks
        :param tracks: list of tracks
        """

        frame_count = sum([1 for t in tracks if t.age >= self.min_age])
        self.total_counts += frame_count
