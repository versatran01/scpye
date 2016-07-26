import os
import cv2

from scpye.processing.binary_cleaner import BinaryCleaner
from scpye.processing.blob_analyzer import BlobAnalyzer
from scpye.utility.data_manager import DataManager
from scpye.utility.visualization import imshow, draw_bboxes

# %%
base_dir = '/home/chao/Workspace/dataset/agriculture'
color = 'red'
mode = 'slow_flash'
side = 'north'
bag_ind = 1

# %%
dm = DataManager(base_dir, color=color, mode=mode, side=side)
image_dir = os.path.join(dm.image_dir, "frame" + str(bag_ind))

i = 46
bw_name = 'bw{0:04d}.png'
bgr_name = 'bgr{0:04d}.png'

bw_file = os.path.join(image_dir, bw_name.format(i))
bgr_file = os.path.join(image_dir, bgr_name.format(i))
bw = cv2.imread(bw_file, cv2.IMREAD_GRAYSCALE)
bgr = cv2.imread(bgr_file, cv2.IMREAD_COLOR)

# %%
bc = BinaryCleaner(ksize=3, iters=2, min_area=5)
bw, region_props = bc.clean(bw)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

disp_bgr = bgr.copy()
disp_bw = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

# %%
ba = BlobAnalyzer()
fruits = ba.analyze(bgr, region_props)

draw_bboxes(disp_bgr, fruits, color=(0, 255, 0))
draw_bboxes(disp_bw, fruits, color=(0, 255, 0))
imshow(disp_bgr, disp_bw, interp='none', figsize=(12, 16))
