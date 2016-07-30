import os
import cv2
import matplotlib.pyplot as plt
from scpye.improc.binary_cleaner import BinaryCleaner
from scpye.improc.blob_analyzer import BlobAnalyzer
from scpye.utils.data_manager import DataManager
from scpye.utils.drawing import imshow, draw_bboxes
from scpye.improc.image_processing import enhance_contrast

# %%
base_dir = '/home/chao/Workspace/dataset/agriculture'
color = 'red'
mode = 'slow_flash'
side = 'north'
bag_ind = 2

# %%
dm = DataManager(base_dir, color=color, mode=mode, side=side)
image_dir = os.path.join(dm.image_dir, "frame" + str(bag_ind))

bw_name = 'bw{0:04d}.png'
bgr_name = 'bgr{0:04d}.png'

plt.ion()
fig = plt.figure(figsize=(12, 16))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
h1, h2 = None, None

for i in range(20, 50):
    bw_file = os.path.join(image_dir, bw_name.format(i))
    bgr_file = os.path.join(image_dir, bgr_name.format(i))
    bw = cv2.imread(bw_file, cv2.IMREAD_GRAYSCALE)
    bgr = cv2.imread(bgr_file, cv2.IMREAD_COLOR)

    # %%
    bc = BinaryCleaner(ksize=3, iters=2, min_area=4)
    bw, region_props = bc.clean(bw)

    disp_bgr = enhance_contrast(bgr)
    disp_bw = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

    # %%
    ba = BlobAnalyzer()
    fruits = ba.analyze(bgr, region_props)

    draw_bboxes(disp_bgr, fruits, color=(0, 255, 0))
    draw_bboxes(disp_bw, fruits, color=(0, 255, 0))
    if h1 is None:
        h1 = ax1.imshow(disp_bgr, interpolation='none')
        h2 = ax2.imshow(disp_bw, interpolation='none')
    else:
        h1.set_data(disp_bgr)
        h2.set_data(disp_bw)
    plt.pause(0.1)
    plt.draw()

