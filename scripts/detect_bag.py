from scpye.data_manager import DataManager
from scpye.fruit_detector import FruitDetector
from scpye.fruit_visualizer import FruitVisualizer

base_dir = '/home/chao/Workspace/dataset/agriculture'
color = 'red'
mode = 'slow_flash'
side = 'north'
bag_ind = 1

dm = DataManager(base_dir, color=color, mode=mode, side=side)
fd = FruitDetector.from_pickle(dm)
fv = FruitVisualizer(dm.image_dir, bag_ind)

# %%
for image in dm.load_bag(bag_ind):
    bgr, bw = fd.detect(image)
    fv.show(bgr, bw)
