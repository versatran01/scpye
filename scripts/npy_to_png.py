import os
from os import listdir
from os.path import join
import numpy as np
import cv2

#npy_dir = '/home/chao/Workspace/dataset/mango_2016/result/mango_v0_2016-09-23-19-24-15/Mango_09-23-19-24-15_Final'
#npy_dir = '/media/chao/Samsung_T3/mango_2016/result/mango_v0_2016-09-23-19-27-50/Mango_09-23-19-27-50_Final'
#npy_paths = [join(npy_dir, f) for f in listdir(npy_dir) if f.endswith('.npy')]
#
#for npy_path in npy_paths:
#    filename, _ = os.path.splitext(npy_path)
#    image = np.load(npy_path)
#    image[image < 0.8] = 0
#    image = np.array(image*255, np.uint8)
#    image_path = filename + '.png'
#    cv2.imwrite(image_path, image)

data_dir = '/media/chao/Samsung_T3/mango_2016/result/mango_v0_2016-09-23-19-27-50'
data_dir = '/media/chao/Samsung_T3/mango_2016/result/mango_v0_2016-09-23-19-24-15'
image_dir = os.path.join(data_dir, 'image')
detect_dir = os.path.join(data_dir, 'detect')
image_paths = [join(image_dir, f) for f in listdir(image_dir) if f.endswith('.png')]

for image_path in image_paths:
    filename, _ = os.path.splitext(image_path)
    image_name = os.path.basename(filename)
    number = image_name[-5:]
    image = cv2.imread(image_path)
    image = cv2.pyrDown(image)
    image = np.rot90(image, 1)
    detect_name = 'detect_color_' + number
    detect_path = os.path.join(detect_dir, detect_name + '.png')
    cv2.imwrite(detect_path, image)
