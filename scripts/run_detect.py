import logging
from scpye.utils.image_dataset import ImageDataset
from scpye.utils.fruit_visualizer import FruitVisualizer

logging.basicConfig(level=logging.INFO)

data_dir = '/home/chao/Workspace/dataset/apple_2016/result/' \
           'apple_v0_mid_density_led_2016-08-24-23-32-50'
ds = ImageDataset(data_dir)

# %%
fd = ds.load_model()
fv = FruitVisualizer()

# %%
for index in range(1087, 1483, 2):
    image = ds.load_image(index)
    bgr, bw = fd.detect(image)
    fv.show(bgr, bw)
    ds.save_detect(index, bgr, bw)
    print(index)
