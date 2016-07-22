import cv2
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d

b = 1
p = 2 * b + 1
image = np.arange(0, 16).reshape((4, 4))
mask = np.eye(4, dtype=bool)
reflect = cv2.copyMakeBorder(image, b, b, b, b, cv2.BORDER_REFLECT)
patches = extract_patches_2d(reflect, (p, p))
patches = np.reshape(patches, (4, 4, -1))

# %%