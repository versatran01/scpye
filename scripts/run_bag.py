from scpye.data_reader import DataReader
from scpye.fruit_detector import FruitDetector
from scpye.blob_analyzer import BlobAnalyzer
from scpye.fruit_tracker import FruitTracker
from scpye.fruit_visualizer import FruitVisualizer

base_dir = '/home/chao/Workspace/bag'
color = 'red'
mode = 'slow_flash'
side = 'south'
bag_ind = 4
min_area = 12

dr = DataReader(base_dir, color=color, mode=mode, side=side)
fd = FruitDetector.from_pickle(dr.model_dir)
ba = BlobAnalyzer(split=False, min_area=min_area)
ft = FruitTracker(min_age=3, max_level=4)
fv = FruitVisualizer(pause_time=0.5)

for image in dr.load_bag(bag_ind):
    bw = fd.detect(image)
    fruits, bw_clean = ba.analyze(bw, fd.v)
    ft.track(fd.color, fruits)
    fv.show(ft.disp, bw_clean)
    print(ft.total_counts)

ft.finish()
print(ft.total_counts)
dr.save_count(bag_ind, ft.frame_counts)
