import numpy as np
from scpye.data_reader import DataReader
from scpye.training import create_image_pipeline, train_image_classifier
from scpye.testing import test_image_classifier

# %%
base_dir = '/home/chao/Workspace/dataset/agriculture'
color = 'red'
mode = 'slow_flash'
side = 'north'

# %%

train_inds = range(0, 12, 3) + range(1, 12, 3)
test_inds = range(2, 12, 3)

k = 0.5
v_min = 28
bbox = np.array([350, 0, 500, 1440])
use_loc = True
cspace = ['hsv', 'lab']
method = 'lr'

# %%
drd = DataReader(base_dir, color=color, mode=mode, side=side)
img_ppl = create_image_pipeline(bbox=bbox, k=k, v_min=v_min, cspace=cspace,
                                use_loc=use_loc)
img_clf = train_image_classifier(drd, train_inds, img_ppl, method=method)
