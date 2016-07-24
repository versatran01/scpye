# %%
import numpy as np
from scpye.data_manager import DataManager
from scpye.pipeline_factory import (create_image_pipeline,
                                    create_feature_pipeline)
from scpye.training import (create_voting_classifier,
                            cross_validate_classifier)

# %%
def train_image_classifier(data_manager, image_indices, image_pipeline,
                           feature_pipeline):
    """
    :type data_reader: DataReader
    :param image_indices: list of indices
    :type image_pipeline: ImagePipeline
    :type feature_pipeline: ImagePipeline
    :rtype: GridSearchCV
    """
    # Load
    Is, Ls = data_manager.load_image_label_list(image_indices)
    # Transform
    Its, Lts = image_pipeline.transform(Is, Ls)
    Xt, yt = feature_pipeline.fit_transform(Its, Lts)
    # Fit
    clf, param_grid = create_voting_classifier()
    grid = cross_validate_classifier(Xt, yt, clf, param_grid)

    return grid


# %%
base_dir = '/home/chao/Workspace/dataset/agriculture'
color = 'red'
mode = 'slow_flash'
side = 'north'

# %%
do_train = False
do_save = True
do_test = True

# %%
if side == 'north':
    train_inds = range(0, 12, 3) + range(1, 12, 3)
    test_inds = range(2, 12, 3)
else:
    train_inds = range(12, 16)
    test_inds = range(12, 16)

# Parameters
k = 0.5
pmin = 26
cspace = ['hsv', 'lab']
loc = True
patch = True
if color == 'red':
    bbox = np.array([300, 0, 600, 1440])
else:
    bbox = np.array([300, 240, 600, 1440])

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
