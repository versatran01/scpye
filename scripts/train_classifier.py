from scpye.detect.pipeline_factory import (create_image_pipeline,
                                           create_feature_pipeline)
from scpye.detect.train_test import (train_image_classifier,
                                     test_image_classifier)
from scpye.utils.data_manager import DataManager

# %%
base_dir = '/home/chao/Workspace/dataset/agriculture'
color = 'red'
mode = 'slow_flash'
side = 'north'

# %%
do_train = True
do_save = True
do_test = True

# %%
if side == 'north':
    train_inds = range(4) + range(8, 12)
    test_inds = range(4, 8)
else:
    train_inds = range(12, 16)
    test_inds = range(12, 16)

# Parameters
k = 0.7
pmin = 27
cspace = ['hsv']
loc = True
patch = True
if color == 'red':
    bbox = [300, 0, 600, 1440]
else:
    bbox = [300, 240, 600, 1440]

# %%
# DataReader
dmg = DataManager(base_dir, color=color, mode=mode, side=side)

if do_train:
    img_ppl = create_image_pipeline(bbox=bbox, k=k)
    ftr_ppl = create_feature_pipeline(pmin=pmin, cspace=cspace, loc=loc,
                                      patch=patch)
    img_clf = train_image_classifier(dmg, train_inds, img_ppl, ftr_ppl)

    if do_save:
        print('saving all models')
        dmg.save_all_models(img_ppl, ftr_ppl, img_clf)

if do_test:
    print('loading all models')
    img_ppl, ftr_ppl, img_clf = dmg.load_all_models()
    test_image_classifier(dmg, test_inds, img_ppl, ftr_ppl, img_clf)
