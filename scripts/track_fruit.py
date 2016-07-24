import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scpye.data_manager import DataManager
from scpye.visualization import imshow
from scpye.blob_analyzer import BlobAnalyzer

base_dir = '/home/chao/Workspace/dataset/agriculture'
color = 'red'
mode = 'slow_flash'
side = 'north'
bag_ind = 1


dm = DataManager(base_dir, color=color, mode=mode, side=side)
image_dir = os.path.join(dm.image_dir, "frame" + str(bag_ind))

i = 150
bw_name = 'bw{0:04d}.png'
bgr_name = 'bgr{0:04d}.png'

bw_file = os.path.join(image_dir, bw_name.format(i))
bgr_file = os.path.join(image_dir, bgr_name.format(i))
bw = cv2.imread(bw_file, cv2.IMREAD_GRAYSCALE)
bgr = cv2.imread(bgr_file, cv2.IMREAD_COLOR)

imshow(bgr, bw, figsize=(12, 16), interp='none', cmap=plt.cm.viridis)

ba = BlobAnalyzer(ksize=3, iters=2)
gray, bw_clean = ba.analyze(bgr, bw)

bw = np.array(bw > 0, dtype=np.uint8)
bw_clean = np.array(bw_clean > 0, dtype=np.uint8)
imshow(bgr, bw + bw_clean, interp='none', cmap=plt.cm.viridis)