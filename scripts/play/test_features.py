import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scpye.visualization import imshow

# %%
image = cv2.imread("/home/chao/Workspace/repo/versatran01/scpye/image/test.png")
image = image[200:-200, :1500, :]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

SCHARR_SCALE = 1 / 16.0
Ix = cv2.Scharr(gray, cv2.CV_32F, 1, 0, scale=SCHARR_SCALE)
Iy = cv2.Scharr(gray, cv2.CV_32F, 0, 1, scale=SCHARR_SCALE)
im_mag, im_ang = cv2.cartToPolar(Ix, Iy)
imshow(im_ang)