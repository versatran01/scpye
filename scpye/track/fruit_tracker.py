from __future__ import (print_function, division, absolute_import)

import logging
from itertools import izip

import cv2
import numpy as np

from scpye.improc.image_processing import enhance_contrast
from scpye.track.assignment import hungarian_assignment
from scpye.track.bounding_box import bboxes_assignment_cost
from scpye.track.fruit_track import FruitTrack
from scpye.track.optical_flow import (calc_optical_flow, calc_average_flow)
from scpye.utils.drawing import (Colors, draw_bboxes, draw_optical_flows,
                                 draw_text)


class FruitTracker(object):
    def __init__(self, min_age=3, win_size=31, max_level=3, init_flow=(40, 0),
                 state_cov=(5, 5, 5, 5), proc_cov=(8, 4, 4, 2),
                 flow_cov=(2, 2), bbox_cov=(1, 1)):
        """
        :param min_age: minimum age of a tracking to be considered for counting
        """
        # Tracking
        self.tracks = []
        self.min_age = min_age
        self.total_counts = 0
        self.frame_counts = 0

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

        self.logger = logging.getLogger(__name__)
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

        # VISUALIZATION: new detection
        # if self.vis:
        #     draw_bboxes(self.disp_bgr, fruits, color=Colors.blue)
        #     draw_bboxes(self.disp_bw, fruits, color=Colors.blue)

        # Initialization
        if not self.initialized:
            self.prev_gray = gray
            self.add_new_tracks(self.tracks, fruits)
            self.logger.info('Tracker initialized.')
            return

        tracks = [t for t in self.tracks]
        # print(len(self.tracks) - len(tracks))

        self.predict_tracks(tracks)
        self.logger.debug("Predicted tracks: {}".format(len(tracks)))

        # VISUALIZATION: after prediction
        # if self.vis:
        #     predict_bboxes = [t.bbox for t in self.tracks]
        #     draw_bboxes(self.disp_bgr, predict_bboxes, color=Colors.red)
        #     draw_bboxes(self.disp_bw, predict_bboxes, color=Colors.red)

        updated_tracks, lost_tracks = self.update_tracks(gray, tracks)
        self.logger.debug("update/lost: {0}/{1}".format(len(updated_tracks),
                                                        len(lost_tracks)))

        # VISUALIZATION: after optical flow update
        # if self.vis:
        #     updated_bboxes = [t.bbox for t in updated_tracks]
        #     draw_bboxes(self.disp_bgr, updated_bboxes, color=Colors.cyan)
        #     draw_bboxes(self.disp_bw, updated_bboxes, color=Colors.cyan)

        matched_tracks, new_fruits, unmatched_tracks = self.match_tracks(
            updated_tracks,
            fruits)
        self.logger.debug("matched/new/unmatched: {0}/{1}/{2}".format(
            len(matched_tracks), len(new_fruits), len(unmatched_tracks)
        ))

        # VISUALIZATION: after hungarian assignment update
        # if self.vis:
        #     matched_bboxes = [t.bbox for t in matched_tracks]
        #     draw_bboxes(self.disp_bgr, matched_bboxes, color=Colors.green)
        #     draw_bboxes(self.disp_bw, matched_bboxes, color=Colors.green)

        # Assemble all lost tracks and update tracks
        self.tracks = matched_tracks
        self.add_new_tracks(self.tracks, new_fruits)

        lost_tracks.extend(unmatched_tracks)
        self.logger.debug(
            "tracks/lost: {0}/{1}".format(len(self.tracks), len(lost_tracks)))

        # VISUALIZATION:
        if self.vis:
            counted_bboxes = [t.bbox for t in self.tracks if
                              t.age >= self.min_age]
            tracked_bboxes = [t.bbox for t in self.tracks if
                              1 < t.age <= self.min_age]
            new_bboxes = [t.bbox for t in self.tracks if t.age == 1]
            if len(counted_bboxes):
                draw_bboxes(self.disp_bgr, counted_bboxes, color=Colors.green,
                            thickness=2)
                draw_bboxes(self.disp_bw, counted_bboxes, color=Colors.green,
                            thickness=2)
            if len(tracked_bboxes):
                draw_bboxes(self.disp_bgr, tracked_bboxes, color=Colors.yellow,
                            thickness=2)
                draw_bboxes(self.disp_bw, tracked_bboxes, color=Colors.yellow,
                            thickness=2)
            if len(new_bboxes):
                draw_bboxes(self.disp_bgr, new_bboxes, color=Colors.red,
                            thickness=2)
                draw_bboxes(self.disp_bw, new_bboxes, color=Colors.red,
                            thickness=2)
        # if self.vis:
        #     for track in self.tracks:
        #         draw_line(self.disp_bgr, track.hist, color=Colors.magenta)
        #         draw_line(self.disp_bw, track.hist, color=Colors.magenta)

        # Count fruits in lost tracks
        self.count_in_tracks(lost_tracks)
        self.logger.info(
            "Frame/Total counts: {0}/{1}".format(self.frame_counts,
                                                 self.total_counts))
        if self.vis:
            h, w = np.shape(bw)
            draw_text(self.disp_bgr, self.total_counts, (10, h - 10), scale=1.5,
                      color=Colors.cyan)
            draw_text(self.disp_bw, self.total_counts, (10, h - 10), scale=1.5,
                      color=Colors.cyan)

    def predict_tracks(self, tracks):
        """
        Predict tracks in Kalman filter
        """
        if np.size(tracks) == 0:
            return
        for track in tracks:
            track.predict()

    def update_tracks(self, gray, tracks):
        """
        Update tracks' position in Kalman filter via KLT
        :param gray: greyscale image
        :param tracks:
        :return: updated tracks, lost_tracks
        """
        # No op when no tracks
        if len(tracks) == 0:
            return [], []

        # Get points in previous image and points in current image
        prev_pts = [t.prev_pos for t in tracks]
        init_pts = [t.pos for t in tracks]

        curr_pts, status = calc_optical_flow(self.prev_gray, gray,
                                             prev_pts, init_pts,
                                             self.win_size,
                                             self.max_level)
        status = np.atleast_1d(status)

        # Update init flow
        if np.sum(status) > 0:
            self.init_flow = calc_average_flow(prev_pts, curr_pts, status)
        self.logger.debug("init flow: {}".format(self.init_flow))

        # VISUALIZATION: optical flow
        draw_optical_flows(self.disp_bgr, prev_pts, curr_pts, status,
                           radius=2, color=Colors.magenta)
        draw_optical_flows(self.disp_bw, prev_pts, curr_pts, status,
                           radius=2, color=Colors.magenta)

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
        :return: matched_tracks, new_fruits, unmatched_tracks
        """
        bboxes_update = np.array([t.bbox for t in tracks])
        bboxes_detect = np.array(fruits)

        # handle cases when tracks or fruits is empty
        if np.size(bboxes_update) != 0 and np.size(bboxes_detect) != 0:
            cost = bboxes_assignment_cost(bboxes_update, bboxes_detect)
            match_inds, lost_inds, new_inds = hungarian_assignment(cost)
        elif np.size(bboxes_update) == 0 and np.size(bboxes_detect) != 0:
            print('no tracks, new detect')
            match_inds = []
            lost_inds = []
            new_inds = np.arange(len(bboxes_detect))
        elif np.size(bboxes_update) != 0 and np.size(bboxes_detect) == 0:
            print('non detect, with tracks')
            match_inds = []
            lost_inds = np.arange(len(bboxes_update))
            new_inds = []
        else:
            print('both are zero')
            match_inds = []
            lost_inds = []
            new_inds = []

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

        # extract new tracks
        if len(new_inds) == 0:
            new_fruits = []
        else:
            new_fruits = fruits[new_inds]

        # get unmatched tracks
        unmatched_tracks = [tracks[ind] for ind in lost_inds]

        return matched_tracks, new_fruits, unmatched_tracks

    def count_in_tracks(self, tracks):
        """
        Count how many fruits there are in tracks
        :param tracks: list of tracks
        """
        self.frame_counts = sum([1 for t in tracks if t.age >= self.min_age])
        self.total_counts += self.frame_counts

    def finish(self):
        self.count_in_tracks(self.tracks)
        self.logger.info("Total counts: {}".format(self.total_counts))
