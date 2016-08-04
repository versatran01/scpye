import matplotlib.pyplot as plt
from scpye.detect.train_test import DetectionModel
from scpye.improc.binary_cleaner import BinaryCleaner
from scpye.improc.blob_analyzer import BlobAnalyzer
from scpye.improc.contour_analysis import analyze_contours_bw
from scpye.utils.data_manager import DataManager
from scpye.utils.bag_manager import BagManager
from scpye.utils.drawing import draw_blob_analyzer, imshow
from scpye.utils.fruit_visualizer import FruitVisualizer
import scipy.ndimage as ndi

# %%
base_dir = '/home/chao/Workspace/dataset/agriculture'
color = 'red'
mode = 'slow_flash'
side = 'north'

# %%
dm = DataManager(base_dir, color=color, mode=mode, side=side)
bc = BinaryCleaner(ksize=5, iters=1)

I, L = dm.load_image_and_label(7)
dp = dm.load_detector()

imshow(Lt, bw, figsize=(12, 16), cmap=plt.cm.viridis)

# %%
labels_true, n_true = ndi.label(Lt)
labels_detect, n_detect = ndi.label(bw)
bw_clean = bc.clean(bw)
labels_clean, n_clean = ndi.label(bw_clean)
print('number of contours in label: ', n_true)
print('number of contours in detect: ', n_detect)
print('number of contours in detect: ', n_clean)
imshow(labels_true, labels_detect, labels_clean, figsize=(15, 18),
       cmap=plt.cm.jet, titles=('a', 'b', 'c'))
