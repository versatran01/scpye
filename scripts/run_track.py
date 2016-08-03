from scpye.improc.binary_cleaner import BinaryCleaner
from scpye.improc.blob_analyzer import BlobAnalyzer

from scpye.track.fruit_tracker import FruitTracker
from scpye.utils.data_manager import DataManager
from scpye.utils.bag_manager import BagManager
from scpye.utils.fruit_visualizer import FruitVisualizer

# %%
base_dir = '/home/chao/Workspace/dataset/agriculture'
color = 'red'
mode = 'fast_flash'
side = 'north'
bag_ind = 4

# %%
dm = DataManager(base_dir, color=color, mode=mode, side=side)
bm = BagManager(dm.data_dir, bag_ind)

bc = BinaryCleaner(ksize=3, iters=2, min_area=16)
ba = BlobAnalyzer(max_cntr_area=180)
ft = FruitTracker(max_level=3)
fv = FruitVisualizer(pause_time=0.1)

# %%
for bgr, bw in bm.load_detect():
    bw_clean, region_props = bc.clean(bw)
    fruits = ba.analyze(bgr, region_props)
    ft.track(bgr, fruits, bw_clean)
#    fv.show(ft.disp_bgr, ft.disp_bw)
    bm.save_track(ft.disp_bgr, ft.disp_bw)
ft.finish()
