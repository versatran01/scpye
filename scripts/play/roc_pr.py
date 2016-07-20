import numpy as np
from scpye.data_reader import DataReader
from scpye.training import create_image_pipeline, train_image_classifier
from scpye.testing import test_image_classifier
from scpye.visualization import imshow
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split

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
cspace = ["hsv", "lab"]
method = 'lr'

# %%
drd = DataReader(base_dir, color=color, mode=mode, side=side)
img_ppl = create_image_pipeline(bbox=bbox, k=k, v_min=v_min, cspace=cspace,
                                use_loc=use_loc)

Is, Ls = drd.load_image_label_list(train_inds)
X_train, y_train = img_ppl.fit_transform(Is, Ls)

# %%
# Train a logistic regression classifier
X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.2)
param_grid = [{'C': [0.1, 1, 10, 100]}]
clf = LogisticRegression(class_weight='balanced')
grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=4, verbose=5,
                    scoring="f1_weighted")
grid.fit(X_t, y_t)

# %%
I, L = drd.load_image_label(2)
X = img_ppl.transform(I)
y = grid.predict_proba(X)
bw = img_ppl.named_steps['remove_dark'].mask
proba = np.array(bw, dtype=np.float)
proba[proba > 0] = y[:, 1]
