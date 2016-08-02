from __future__ import (print_function, division, absolute_import)

import os
import logging

import cv2
import rosbag
from cv_bridge import CvBridge, CvBridgeError


class BagManager(object):
    def __init__(self, data_dir, index, bag='bag', detect='detect',
                 track='track'):
        self.data_dir = data_dir
        self.index = index

        self.bag_fmt = "frame{0}.bag"
        self.bgr_fmt = "bgr{0:04d}.png"
        self.bw_fmt = "bw{0:04d}.png"
        self.i_detect = 0
        self.i_track = 0

        self.bag_dir = os.path.join(self.data_dir, bag)
        self.detect_dir = os.path.join(self.bag_dir, detect,
                                       "frame{0}".format(index))
        self.track_dir = os.path.join(self.bag_dir, track,
                                      "frame{0}".format(index))

        self.logger = logging.getLogger(__name__)
        self.logger.info("BagManger: {}".format(self.bag_dir))
        self.logger.debug("test debug")

    def load_bag(self, topic='/color/image_rect_color'):
        """
        A generator for image
        :param topic: image message topic
        :return:
        """
        bagname = os.path.join(self.bag_dir,
                               self.bag_fmt.format(self.index))
        self.logger.info('loading bag: {0}'.format(bagname))

        bridge = CvBridge()
        with rosbag.Bag(bagname) as bag:
            for topic, msg, t in bag.read_messages(topic):
                try:
                    image = bridge.imgmsg_to_cv2(msg)
                except CvBridgeError as e:
                    print(e)
                    continue
                yield image

    def save_detect(self, bgr, bw):
        self.logger.info('saving image {}', self.i_detect)
        bgr_name = os.path.join(self.detect_dir,
                                self.bgr_fmt.format(self.i_detect))
        bw_name = os.path.join(self.detect_dir,
                               self.bw_fmt.format(self.i_detect))
        cv2.imwrite(bgr_name, bgr)
        cv2.imwrite(bw_name, bw)
        self.i_detect += 1

        self.logger.debug("save bgr: {}".format(bgr_name))
        self.logger.debug("save bw: {}".format(bw_name))

    def load_detect(self):
        i = 0
        while True:
            bgr_name = os.path.join(self.detect_dir,
                                    self.bgr_fmt.format(i))
            bw_name = os.path.join(self.detect_dir,
                                   self.bw_fmt.format(i))
            bgr = cv2.imread(bgr_name, cv2.IMREAD_COLOR)
            bw = cv2.imread(bw_name, cv2.IMREAD_GRAYSCALE)

            self.logger.debug("load bgr: {}".format(bgr_name))
            self.logger.debug("load bw: {}".format(bw_name))

            if bgr is None or bw is None:
                self.logger.debug("No image left at {}".format(i))
                break
            else:
                i += 1
                yield bgr, bw

    def save_track(self, disp_bgr, disp_bw):
        self.logger.info('saving image', self.i_track)

        bgr_name = os.path.join(self.track_dir,
                                self.bgr_fmt.format(self.i_track))
        bw_name = os.path.join(self.track_dir,
                               self.bw_fmt.format(self.i_track))
        cv2.imwrite(bgr_name, disp_bgr)
        cv2.imwrite(bw_name, disp_bw)
        self.i_track += 1
