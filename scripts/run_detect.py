from scpye.utils.image_dataset import ImageDataset
from scpye.utils.fruit_visualizer import FruitVisualizer
import os

base_dir = '/home/chao/Workspace/dataset/apple_2016/result/'
#data_name = 'apple_v0_mid_density_led_2016-08-24-23-32-50'
data_name = 'apple_v0_high_density_led_2016-08-25-23-38-10'
data_dir = os.path.join(base_dir, data_name)

print(data_name)
ds = ImageDataset(data_dir)

# %%
fd = ds.load_model()
fv = FruitVisualizer()

# %%
for index in range(5178, 5264, 2):
    image = ds.load_image(index)
    bgr, bw = fd.detect(image)
#    fv.show(bgr, bw)
    ds.save_detect(index, bgr, bw)
    print(index)
