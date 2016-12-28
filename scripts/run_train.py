# -*- coding: utf-8 -*-
from scpye.utils.image_dataset import ImageDataset
from scpye.detect.pipeline_factory import (create_image_pipeline,
                                           create_feature_pipeline)
from scpye.detect.train_test import (train_fruit_detector,
                                     test_fruit_detector)
import os
# %%
base_dir = '/home/chao/Workspace/dataset/apple_2016/result'
#data_name = 'apple_v0_mid_density_led_2016-08-24-23-32-50'
#data_name = 'apple_v0_high_density_led_2016-08-25-23-38-10'
data_name = 'apple_v0_low_density_led_2016-08-23-23-36-16'
data_dir = os.path.join(base_dir, data_name)
ds = ImageDataset(data_dir)
Is, Ls = ds.load_train()
n = 9

# %%
# Parameters
k = 0.4
pmin = 27
cspace = ['hsv']
loc = True
patch = False
grad = True
bbox = [300, 0, 1000, 2400]

# %%
do_train = True
do_save = True
do_test = False

# %%
# Train
if do_train:
    img_ppl = create_image_pipeline(bbox=bbox, k=k)
    ftr_ppl = create_feature_pipeline(pmin=pmin, cspace=cspace, loc=loc,
                                      patch=patch, grad=grad)
    fd = train_fruit_detector(Is[:n], Ls[:n], img_ppl, ftr_ppl)

# %%
# Save
if do_save:
    ds.save_model(fd)

# %%
# Test
if do_test:
    import matplotlib.pyplot as plt

    test_fruit_detector(Is[n:], Ls[n:], fd)
    plt.show()
