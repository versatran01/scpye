from scpye.detect.fruit_detector import FruitDetector
from scpye.utils.data_manager import DataManager
from scpye.utils.fruit_visualizer import FruitVisualizer

base_dir = '/home/chao/Workspace/dataset/agriculture'
color = 'red'
mode = 'slow_flash'
side = 'north'
bag_ind = 2

dm = DataManager(base_dir, color=color, mode=mode, side=side)
fd = FruitDetector.from_pickle(dm)
fv = FruitVisualizer(dm.image_dir, bag_ind, save_image=True)

# %%
for image in dm.load_bag(bag_ind):
    bgr, bw = fd.detect(image)
    fv.show(bgr, bw)
