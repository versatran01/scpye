from __future__ import (print_function, division, absolute_import)

import os
import cv2
import numpy as np
from sklearn.externals import joblib
from scpye.image_pipeline import ImagePipeline


class FruitDetector(object):
    fruit_dtype = [('bbox', np.int, 4), ('num', np.int, 1)]

    def __init__(self, img_ppl, img_clf):
        """
        :param img_ppl: image pipeline
        :type img_ppl: ImagePipeline
        :param img_clf: classifier
        """
        self.img_ppl = img_ppl
        self.img_clf = img_clf
        self.bw = None

    @property
    def color(self):
        return self.img_ppl.named_steps['remove_dark'].bgr.copy()

    @property
    def gray(self):
        return cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY)

    @property
    def v(self):
        return self.img_ppl.named_steps['remove_dark'].hsv[:, :, -1].copy()

    def detect(self, image):
        Xt = self.img_ppl.transform(image)
        y = self.img_clf.predict(Xt)
        bw = np.array(self.img_ppl.named_steps['remove_dark'].mask, copy=True)
        bw[bw > 0] = y
        return bw

    @classmethod
    def from_pickle(cls, model_dir):
        """
        Constructor from a pickle
        :param model_dir:
        :return:
        :rtype: FruitDetector
        """
        img_ppl_file = os.path.join(model_dir, 'img_ppl.pkl')
        img_clf_file = os.path.join(model_dir, 'img_clf.pkl')
        img_ppl = joblib.load(img_ppl_file)
        img_clf = joblib.load(img_clf_file)
        return cls(img_ppl, img_clf)
