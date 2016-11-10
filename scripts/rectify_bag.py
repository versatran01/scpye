# -*- coding: utf-8 -*-
from __future__ import (print_function, absolute_import, division)
import os
import rosbag
import cv2
import numpy as np
from tqdm import tqdm
from image_geometry.cameramodels import PinholeCameraModel
from cv_bridge import CvBridge

# %%
data_dir = '/home/chao/Workspace/dataset/apple_2016'
bag_dir = os.path.join(data_dir, 'bag')
bag_name = 'apple_v0_mid_density_led_2016-08-24-23-32-50.bag'
bag_file = os.path.join(bag_dir, bag_name)
result_dir = os.path.join(data_dir, 'result')
image_dir_name = os.path.splitext(bag_name)[0]
image_dir = os.path.join(result_dir, image_dir_name, 'image')

if not os.path.exists(image_dir):
    os.makedirs(image_dir)

print(bag_file)
print(image_dir)

# %%
camera_topic = '/color'
image_topic = camera_topic + '/image_raw'
cinfo_topic = camera_topic + '/camera_info'
image_name_fmt = 'image_rect_color_{0:05}.png'

bridge = CvBridge()
cam_model = PinholeCameraModel()

with rosbag.Bag(bag_file) as bag:
    for topic, msg, t in tqdm(bag.read_messages()):
        # first initialize camera model
        if cam_model.K is None and topic == cinfo_topic:
            cam_model.fromCameraInfo(msg)
            print('camera model initialized')

        # after camera model initialized, read image
        if cam_model.K is not None and topic == image_topic:
            image = bridge.imgmsg_to_cv2(msg, 'bgr8')
            image_rect = np.empty_like(image)
            cam_model.rectifyImage(image, image_rect)
            image_name = image_name_fmt.format(msg.header.seq)
            image_file = os.path.join(image_dir, image_name)
            cv2.imwrite(image_file, image_rect)
