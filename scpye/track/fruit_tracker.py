from __future__ import (print_function, division, absolute_import)

import cv2
import numpy as np
from itertools import izip

from scpye.track.assignment import hungarian_assignment
from scpye.track.bounding_box import bboxes_assignment_cost
from scpye.track.fruit_track import FruitTrack
from scpye.track.optical_flow import calc_optical_flow


class FruitTracker(object):
    def __init__(self, min_age=3, win_size=31, max_level=3, init_flow=(40, 0),
                 proc_cov=(5, 2, 5, 2), flow_cov=(1, 1), bbox_cov=(2, 2)):
        """
        :param min_age: minimum age of a tracking to be considered for counting
        """
        # Tracking
        self.tracks = []
        self.min_age = min_age
        self.total_counts = 0

        self.prev_gra = None
        # Optical flow parameters
        self.win_size = win_size
        self.max_level = max_level

        # Kalman filter parameters
        self.init_flow = np.array(init_flow)
        self.proc_cov = np.array(proc_cov)
        self.flow_cov = np.array(flow_cov)
        self.bbox_cov = np.array(bbox_cov)

        # self.disp = None

    @property
    def initialized(self):
        return self.prev_gra is not None

    def add_new_tracks(self, tracks, fruits):
        """
        Add new fruits to tracks
        :param tracks: a list of tracks
        :param fruits: a list of fruits
        """
        for fruit in fruits:
            track = FruitTrack(fruit, self.init_flow, self.proc_cov)
            tracks.append(track)

    def track(self, image, fruits):
        """
        Main tracking step
        :param image: gray scale image
        :param fruits: new fruits
        """
        # Convert to greyscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        disp = image.copy()

        if not self.initialized:
            self.prev_gra = gray
            self.add_new_tracks(self.tracks, fruits)
            return

        # Kalman filter prediction
        self.predict_tracks()

        # Optical flow update
        updated_tracks, lost_tracks = self.update_tracks(gray)

        # Hungarian assignment
        matched_tks, unmatched_tks = self.match_tracks(updated_tracks, fruits)

        # Update matched tracks
        self.tracks = matched_tks

        # Assemble all lost tracks
        lost_tracks.extend(unmatched_tks)

        # Count fruits in invalid tracks
        # self.count_in_tracks(lost_tks)

    def predict_tracks(self):
        """
        Predict tracks in Kalman filter
        Any track that is outside of image by some amount is deemed lost tracks
        """
        for track in self.tracks:
            track.predict()

    def update_tracks(self, gray):
        """
        Update tracks in Kalman filter via KLT
        :param gray:
        :return: updated tracks, invalid_tracks
        """
        updated_tracks, lost_tracks = [], []
        prev_pts = [t.prev_pos for t in self.tracks]
        init_pts = [t.pos for t in self.tracks]
        curr_pts, status = calc_optical_flow(self.prev_gra, gray,
                                             prev_pts, init_pts,
                                             self.win_size,
                                             self.max_level)

        # Remove lost tracks
        for track, point, stat in izip(self.tracks, curr_pts, status):
            if stat:
                track.correct_flow(point, self.flow_cov)
                updated_tracks.append(track)
            else:
                lost_tracks.append(track)

        # Update gray_prev
        self.prev_gra = gray

        return updated_tracks, lost_tracks

    def match_tracks(self, tracks, fruits):
        """
        Match tracks to new detection
        :param tracks:
        :param fruits:
        :return:
        """
        bboxes_update = np.array([t.bbox for t in tracks])
        bboxes_detect = np.array(fruits)

        cost = bboxes_assignment_cost(bboxes_update, bboxes_detect)
        match_inds, lost_inds, new_inds = hungarian_assignment(cost)

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


        # def finish(self):
        #     """
        #     Count what's left in tracks, call after final image
        #     """
        #     self.count_in_tracks(self.tracks)
        #     self.frame_counts = np.array(self.frame_counts)

        # def count_in_tracks(self, tracks):
        #     """
        #     Count how many fruits there are in tracks
        #     :param tracks: list of tracks
        #     """
        #     temp_sum = 0
        #     for track in tracks:
        #         if track.age >= self.min_age:
        #             temp_sum += track.num
        #
        #     self.frame_counts.append(temp_sum)
        #     self.total_counts += temp_sum
        #
        #     # ===== DRAW TOTAL COUNTS =====
        #     draw_text(self.disp, self.total_counts, (0, len(self.disp) - 5),
        #               scale=1, color=Colors.counted)
