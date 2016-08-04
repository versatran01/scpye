import logging

from scpye.detect.pipeline_factory import (create_image_pipeline,
                                           create_feature_pipeline)
from scpye.detect.train_test import (train_image_classifier,
                                     test_image_classifier)
from scpye.utils.data_manager import DataManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %%
# DataManager
base_dir = '/home/chao/Workspace/dataset/agriculture'
color = 'red'
mode = 'slow_flash'
side = 'north'
dmg = DataManager(base_dir, color=color, mode=mode, side=side)

# %%
do_train = True
do_save = True
do_test = True

# %%
if side == 'north':
    train_inds = range(0, 12, 3) + range(2, 12, 3)
    test_inds = range(1, 12, 3)
else:
    train_inds = range(12, 16)
    test_inds = range(12, 16)

# %%
# Parameters
k = 0.5
pmin = 27
cspace = ['hsv']
loc = True
patch = True
if color == 'red':
    bbox = [300, 0, 600, 1440]
else:
    bbox = [300, 220, 600, 1440]

# %%
# Train
if do_train:
    logger.info('start training')
    img_ppl = create_image_pipeline(bbox=bbox, k=k)
    ftr_ppl = create_feature_pipeline(pmin=pmin, cspace=cspace, loc=loc,
                                      patch=patch)
    img_clf = train_image_classifier(dmg, train_inds, img_ppl, ftr_ppl)

    if do_save:
        logger.info('saving all models')
        dmg.save_all(img_ppl, ftr_ppl, img_clf)

# %%
# Test
if do_test:
    import matplotlib.pyplot as plt

    logger.info('loading all models')
    dct_mdl = dmg.load_model()
    test_image_classifier(dmg, test_inds, dct_mdl)
    plt.show()
