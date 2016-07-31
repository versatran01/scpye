from __future__ import (print_function, division, absolute_import)

import os
import cv2
import rosbag
from cv_bridge import CvBridge, CvBridgeError


class BagManager(object):
    def __init__(self, data_dir, index, bag='bag', detect='detect',
                 video='video'):
        self.data_dir = data_dir
        self.index = index

        self.bag_fmt = "frame{0}.bag"
        self.bgr_fmt = "bgr{0:04d}.png"
        self.bw_fmt = "bw{0:04d}.png"
        self.i = 0

        self.bag_dir = os.path.join(self.data_dir, bag)
        self.detect_dir = os.path.join(self.bag_dir, detect,
                                       "frame{0}".format(index))
        self.video_dir = os.path.join(self.bag_dir, video,
                                      "frame{0}".format(index))

    def load_bag(self, topic='/color/image_rect_color'):
        """
        A generator for image
        :param topic: image message topic
        :return:
        """
        bagname = os.path.join(self.bag_dir,
                               self.bag_fmt.format(self.index))
        print('loading bag: {0}'.format(bagname))
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
        print('saving image', self.i)
        bgr_name = os.path.join(self.detect_dir,
                                self.bgr_fmt.format(self.i))
        bw_name = os.path.join(self.detect_dir,
                               self.bw_fmt.format(self.i))
        cv2.imwrite(bgr_name, bgr)
        cv2.imwrite(bw_name, bw)
        self.i += 1

    def load_detect(self):
        i = 0
        while True:
            bgr_name = os.path.join(self.detect_dir,
                                    self.bgr_fmt.format(i))
            bw_name = os.path.join(self.detect_dir,
                                   self.bw_fmt.format(i))
            bgr = cv2.imread(bgr_name, cv2.IMREAD_COLOR)
            bw = cv2.imread(bw_name, cv2.IMREAD_GRAYSCALE)
            if bgr is None or bw is None:
                break
            else:
                i += 1
                yield bgr, bw
