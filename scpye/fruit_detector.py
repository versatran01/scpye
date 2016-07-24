from __future__ import (print_function, division, absolute_import)

import os
import cv2
import numpy as np
from sklearn.externals import joblib
from scpye.image_pipeline import ImagePipeline


def get_dark_remover(feature_pipeline):
    return feature_pipeline.named_steps['remove_dark']


class FruitDetector(object):
    fruit_dtype = [('bbox', np.int, 4), ('num', np.int, 1)]

    def __init__(self, img_ppl, ftr_ppl, img_clf):
        """
        :type img_ppl: ImagePipeline
        :type ftr_ppl: ImagePipeline
        :param img_clf: classifier
        """
        self.img_ppl = img_ppl
        self.ftr_ppl = ftr_ppl
        self.img_clf = img_clf

    @property
    def dark_remover(self):
        return self.ftr_ppl.named_steps['remove_dar']

    @property
    def bgr(self):
        return self.dark_remover.bgr.copy()

    @property
    def gray(self):
        return self.dark_remover.gray.copy()

    @property
    def mask(self):
        return self.dark_remover.mask.copy()

    def detect(self, image):
        It = self.img_ppl.transform(image)
        Xt = self.ftr_ppl.transform(It)
        y = self.img_clf.predict(Xt)

        bw = self.mask
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
