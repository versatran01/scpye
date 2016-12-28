import os
from scpye.improc.binary_cleaner import BinaryCleaner
from scpye.improc.blob_analyzer import BlobAnalyzer
from scpye.track.fruit_tracker import FruitTracker
from scpye.utils.image_dataset import ImageDataset
from scpye.utils.fruit_visualizer import FruitVisualizer

# %%
base_dir = '/home/chao/Workspace/dataset/mango_2016/result'
#data_name = 'apple_v0_mid_density_led_2016-08-24-23-32-50'
#data_name = 'apple_v0_mid_density_led_2016-08-24-23-36-06'
#data_name = 'apple_v0_high_density_led_2016-08-25-23-38-10'
#data_name = 'mango_v0_2016-09-23-19-24-15'
data_name = 'mango_v0_2016-09-23-19-27-50'

print(data_name)
data_dir = os.path.join(base_dir, data_name)
ds = ImageDataset(data_dir)

# %%
bc = BinaryCleaner(ksize=5, iters=1)
ba = BlobAnalyzer()
ft = FruitTracker()
fv = FruitVisualizer(pause_time=0.1)

# %%
for index in range(8590, 8716, 1):
    bgr, bw = ds.load_detect(index)
    bw = bc.clean(bw)
    fruits, bw = ba.analyze(bgr, bw)
    ft.track(bgr, fruits, bw)
    #fv.show(ft.disp_bgr, ft.disp_bw)
    ds.save_track(index, ft.disp_bgr, ft.disp_bw)
    print(index)

ft.finish()
print('final count', ft.total_counts)
