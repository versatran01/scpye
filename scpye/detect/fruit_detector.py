from __future__ import (print_function, division, absolute_import)

from scpye.improc.image_processing import u8_from_bw
import joblib


def get_dark_remover(feature_pipeline):
    return feature_pipeline.named_steps['remove_dark']


class FruitDetector(object):
    def __init__(self, img_ppl, ftr_ppl, img_clf):
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
        bw = self._detect(It, bw_val=255)

        return It, bw

    def detect_image_label(self, image, label):
        """
        Detect fruit on image and transform label
        :param image:
        :param label:
        :return:
        """
        It, Lt = self.img_ppl.transform(image, label)
        bw = self._detect(It, bw_val=1)
        return It, Lt, bw

    def _detect(self, image_transformed, bw_val=255):
        Xt = self.ftr_ppl.transform(image_transformed)
        y = self.img_clf.predict(Xt)

        bw = self.ftr_ppl.named_steps['remove_dark'].mask.copy()
        bw[bw > 0] = y
        bw = u8_from_bw(bw, val=bw_val)
        return bw

    @classmethod
    def from_pickle(cls, path):
        """
        Constructor from a pickle
        """
        fd = joblib.load(path)
        return cls(fd.img_ppl, fd.ftr_ppl, fd.img_clf)

    def to_pickle(self, path):
        joblib.dump(self, path)
