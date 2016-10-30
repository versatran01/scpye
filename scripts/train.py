# -*- coding: utf-8 -*-
from scpye.utils.image_dataset import TrainingSet
from scpye.detect.pipeline_factory import (create_image_pipeline,
                                           create_feature_pipeline)
from scpye.detect.train_test import (train_fruit_detector,
                                     test_fruit_detector)

# %%
data_dir = '/home/chao/Workspace/dataset/apple_2016/result/train'
image_name = 'image_rect_color'
label_name = 'image_rect_label'
ts = TrainingSet(data_dir, image_name, label_name)
Is, Ls = ts.load_image_label_list()
n = 4

# %%
# Parameters
k = 0.25
pmin = 27
cspace = ['hsv']
loc = True
patch = True
grad = True
bbox = None

# %%
do_train = True
do_save = False
do_test = True

# %%
# Train
if do_train:
    img_ppl = create_image_pipeline(bbox=bbox, k=k)
    ftr_ppl = create_feature_pipeline(pmin=pmin, cspace=cspace, loc=loc,
                                      patch=patch, grad=grad)
    fd = train_fruit_detector(Is[:n], Ls[:n], img_ppl, ftr_ppl)

# %%
if do_test:
    import matplotlib.pyplot as plt

    test_fruit_detector(Is[n:n+1], Ls[n:n+1], fd)
    plt.show()
