# -*- coding: utf-8 -*-
from scpye.utils.image_dataset import TrainingSet
from scpye.detect.pipeline_factory import (create_image_pipeline,
                                           create_feature_pipeline)
from scpye.detect.train_test import (train_fruit_detector,
                                     test_fruit_detector)
import os

# %%
data_dir = '/home/chao/Workspace/dataset/apple_2016/result'
train_dir = os.path.join(data_dir, 'train')
model_dir = os.path.join(data_dir, 'model')
model_name = os.path.join(model_dir, 'red.pkl')
image_name = 'image_rect_color'
label_name = 'image_rect_label'
ts = TrainingSet(train_dir, image_name, label_name)
Is, Ls = ts.load_image_label_list()
n = 6

# %%
# Parameters
k = 0.5
pmin = 27
cspace = ['hsv']
loc = True
patch = False
grad = True
bbox = [300, 0, 1000, 2400]

# %%
do_train = True
do_save = True
do_test = True

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
    fd.to_pickle(model_name)

# %%
# Test
if do_test:
    import matplotlib.pyplot as plt

    test_fruit_detector(Is[n:], Ls[n:], fd)
    plt.show()
