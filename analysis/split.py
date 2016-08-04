from scpye.detect.fruit_detector import FruitDetector
from scpye.improc.binary_cleaner import BinaryCleaner
from scpye.improc.blob_analyzer import BlobAnalyzer
from scpye.utils.data_manager import DataManager
from scpye.utils.drawing import imshow
from scpye.improc.image_processing import (enhance_contrast, gray_from_bgr,
                                           bgr_from_gray, swap_channels,
                                           u8_from_bw)
from scpye.utils.drawing import *
import numpy as np

# %%
base_dir = '/home/chao/Workspace/dataset/agriculture'
color = 'red'
mode = 'slow_flash'
side = 'north'

# %%
dm = DataManager(base_dir, color=color, mode=mode, side=side)
fd = FruitDetector.from_pickle(dm)
bc = BinaryCleaner(ksize=5, iters=1)
ba = BlobAnalyzer()

# %%
I, L = dm.load_image_and_label(3)
It, Lt, bw = fd.detect_image_label(I, L)
bw = bc.clean(bw)

# %%
fruits, bw_filled = ba.analyze(It, bw)

# %%
imshow(ba.disp_bgr, ba.disp_bw, figsize=(15, 18))
