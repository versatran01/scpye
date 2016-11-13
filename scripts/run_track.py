import logging
from tqdm import tqdm
from scpye.improc.binary_cleaner import BinaryCleaner
from scpye.improc.blob_analyzer import BlobAnalyzer
from scpye.track.fruit_tracker import FruitTracker
from scpye.utils.image_dataset import ImageDataset
from scpye.utils.fruit_visualizer import FruitVisualizer

# %%
data_dir = '/home/chao/Workspace/dataset/apple_2016/result/' \
           'apple_v0_mid_density_led_2016-08-24-23-32-50'
ds = ImageDataset(data_dir)


# %%
fd = ds.load_model()
bc = BinaryCleaner(ksize=5, iters=1)
ba = BlobAnalyzer()
ft = FruitTracker()
fv = FruitVisualizer(pause_time=0.1)

# %%
for index in range(1087, 1483, 2):
    bgr, bw = ds.load_detect(index)
    bw = bc.clean(bw)
    fruits, bw = ba.analyze(bgr, bw)
    ft.track(bgr, fruits, bw)
    fv.show(ft.disp_bgr, ft.disp_bw)
    # bm.save_track(ft.disp_bgr, ft.disp_bw, save_disp=True)

ft.finish()
