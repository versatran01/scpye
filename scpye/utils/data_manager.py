import os
import logging

import cv2
import numpy as np
from sklearn.externals import joblib

from scpye.utils.exception import ImageNotFoundError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_binary(data):
    return np.array(data > 0, dtype='uint8')


class DataManager(object):
    def __init__(self, base_dir='/home/chao/Workspace/bag', fruit='apple',
                 color='red', mode='slow_flash', side='north', train='train',
                 label='label', model='model'):
        self.base_dir = base_dir
        self.file_fmt = 'frame{0:04d}_{1}.png'

        # Directory
        # data_dir will be passed to BagManger
        self.data_dir = os.path.join(self.base_dir, fruit, color, mode, side)
        self.train_dir = os.path.join(self.data_dir, train)
        self.label_dir = os.path.join(self.train_dir, label)
        self.model_dir = os.path.join(self.train_dir, model)

        logger.info("DataManger: {}".format(self.data_dir))

    def _read_image(self, index, suffix, color=True):
        """
        Read image
        :param index: index of image
        :param suffix: suffix of image
        :param color: color or gray
        :return: image
        :rtype: numpy.ndarray
        """
        filename = os.path.join(self.label_dir,
                                self.file_fmt.format(index, suffix))
        if color:
            flag = cv2.IMREAD_COLOR
        else:
            flag = cv2.IMREAD_GRAYSCALE

        image = cv2.imread(filename, flag)

        if image is None:
            raise ImageNotFoundError(filename)

        return image

    def load_image(self, index):
        """
        Load color image by index
        :param index:
        :return: color image
        """
        logger.debug("Loading image {}".format(index))
        return self._read_image(index, 'raw', color=True)

    def load_label(self, index):
        """
        Load labels by index
        :param index:
        :return: label in bool
        """
        logger.debug("Loading label {}".format(index))
        neg = self._read_image(index, 'neg', color=False)
        pos = self._read_image(index, 'pos', color=False)
        label = np.dstack((neg, pos))
        return make_binary(label)

    def load_image_and_label(self, index):
        """
        :param index: index of image
        :return: image and label
        """
        image = self.load_image(index)
        label = self.load_label(index)
        return image, label

    def save_model(self, model, name, compress=3):
        """
        Save model to model directory
        :param model:
        :param name:
        :param compress: compression level, 3 is recommended
        :return:
        """
        model_pickle = os.path.join(self.model_dir, name + '.pkl')
        joblib.dump(model, model_pickle, compress=compress)
        logger.info('{0} saved to {1}'.format(name, model_pickle))

    def save_all_models(self, img_ppl, ftr_ppl, img_clf):
        self.save_model(img_ppl, 'img_ppl')
        self.save_model(ftr_ppl, 'ftr_ppl')
        self.save_model(img_clf, 'img_clf')

    def load_model(self, name):
        """
        Load model from model directory
        :param name:
        :return:
        """
        model_pkl = os.path.join(self.model_dir, name + '.pkl')
        model = joblib.load(model_pkl)
        logger.info('{0} load from {1}'.format(name, model_pkl))
        return model

    def load_all_models(self):
        img_ppl = self.load_model('img_ppl')
        ftr_ppl = self.load_model('ftr_ppl')
        img_clf = self.load_model('img_clf')
        return img_ppl, ftr_ppl, img_clf

    def load_image_label_list(self, image_indices):
        """
        Load image and label in separate lists
        :param image_indices:
        """
        image_indices = np.atleast_1d(image_indices)

        Is, Ls = [], []
        for ind in image_indices:
            I, L = self.load_image_and_label(ind)
            Is.append(I)
            Ls.append(L)

        return Is, Ls
