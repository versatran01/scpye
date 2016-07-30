from __future__ import (print_function, division, absolute_import)

import numpy as np

from scpye.detect.image_pipeline import ImagePipeline


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

    def detect(self, image):
        """
        Detect fruit on image
        :param image:
        :return: (bgr, bw)
        """
        It = self.img_ppl.transform(image)
        Xt = self.ftr_ppl.transform(It)
        y = self.img_clf.predict(Xt)

        bw = self.ftr_ppl.named_steps['remove_dark'].mask.copy()
        bw[bw > 0] = y
        return It, bw

    @classmethod
    def from_pickle(cls, data_manager):
        """
        Constructor from a pickle
        :type data_manager: DataManager
        :rtype: FruitDetector
        """
        img_ppl, ftr_ppl, img_clf = data_manager.load_all_models()
        return cls(img_ppl, ftr_ppl, img_clf)
