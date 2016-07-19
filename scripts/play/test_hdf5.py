from __future__ import print_function
import h5py as h5
import numpy as np
import cv2
import matplotlib.pyplot as plt

f = h5.File("apple.hdf5")
train_group = f.create_group("train")
