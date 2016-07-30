# %%
import matplotlib.pyplot as plt
import numpy as np

from scpye.detect.pipeline_factory import (create_image_pipeline,
                                           create_feature_pipeline)
from scpye.detect.training import (create_voting_classifier,
                                   cross_validate_classifier)
from scpye.utils.data_manager import DataManager
from scpye.utils.drawing import imshow


# %%
def train_image_classifier(data_manager, image_indices, image_pipeline,
                           feature_pipeline):
    """
    :type data_manager: DataReader
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


def test_image_classifier(data_manager, image_indices, image_pipeline,
                          feature_pipeline, image_classifier):
    """
    :type data_manager: DataManager
    :param image_indices:
    :type image_pipeline: ImagePipeline
    :type feature_pipeline: ImagePipeline
    :type image_classifier: GridSearchCV
    """
    if np.isscalar(image_indices):
        image_indices = [image_indices]

    for ind in image_indices:
        I, L = data_manager.load_image_label(ind)
        It, Lt = image_pipeline.transform(I, L[..., 1])
        Xt = feature_pipeline.transform(It)
        y = image_classifier.predict(Xt)

        bw = feature_pipeline.named_steps['remove_dark'].mask.copy()
        bw[bw > 0] = y
        bw = np.array(bw, dtype='uint8')
        imshow(It, bw + Lt, cmap=plt.cm.viridis)


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
    test_inds = np.arange(4, 8)
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
    test_image_classifier(dmg, test_inds, img_ppl, ftr_ppl, img_clf)
