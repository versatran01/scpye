import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from scpye.data_reader import DataReader
from scpye.image_transformer import ImageRotator
from scpye.image_pipeline import ImagePipeline
from scpye.training import (create_image_pipeline, create_feature_pipeline,
                            train_image_classifier)
from scpye.testing import test_image_classifier
from scpye.visualization import imshow
from scpye.bounding_box import extract_bbox

# %%
base_dir = '/home/chao/Workspace/dataset/agriculture'
color = 'red'
mode = 'slow_flash'
side = 'north'

# %%

train_inds = range(0, 12, 3) + range(1, 12, 3)
test_inds = range(2, 12, 3)

k = 0.5
pmin = 28
bbox = np.array([350, 0, 500, 1440])
loc = True
cspace = ["hsv", "lab"]
method = 'lr'

# %%
drd = DataReader(base_dir, color=color, mode=mode, side=side)
img_ppl = create_image_pipeline(bbox=bbox, k=k)
ftr_ppl = create_feature_pipeline(pmin=pmin, cspace=cspace, loc=loc,
                                  patch=True)

Is, Ls = drd.load_image_label_list(train_inds)
Its, Lts = img_ppl.transform(Is, Ls)
Xt, yt = ftr_ppl.fit_transform(Its, Lts)

# %%
# Train a logistic regression classifier
X_t, X_v, y_t, y_v = train_test_split(Xt, yt, test_size=0.2)
#param_grid = [{'C': [0.1, 1, 10, 100]}]
#clf = LogisticRegression(class_weight='balanced')
#clf = SVC()
#grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=4, verbose=5,
#                    scoring="f1_weighted")
#grid = GaussianNB()
grid.fit(X_t, y_t)

# %%
I, L = drd.load_image_label(8)

X = img_ppl.transform(I)
X = ftr_ppl.transform(X)
y = img_ppl.transform(L[..., 1])

mask = ftr_ppl.named_steps['remove_dark'].mask

y_pred = grid.predict(X)
bw = mask.copy()
bw[bw > 0] = y_pred

y_proba = grid.predict_proba(X)
proba = np.array(mask, dtype=np.float)
proba[proba > 0] = y_proba[:, 1]

# %%
# Plot pr curve
#y_true = y.ravel()
#probas_pred = proba.ravel()
#precision, recall, thresholds = precision_recall_curve(y_true, probas_pred,
#                                                       pos_label=1)
#plt.plot(recall, precision)
# Plot roc/auc curve
