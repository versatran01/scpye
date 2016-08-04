import os
import logging

import cv2
import numpy as np
from sklearn.externals import joblib

from scpye.detect.train_test import DetectionModel
from scpye.utils.exception import ImageNotFoundError


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

        self.logger = logging.getLogger(__name__)
        self.logger.info("DataManger: {}".format(self.data_dir))

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
        self.logger.debug("Loading image {}".format(index))
        return self._read_image(index, 'raw', color=True)

    def load_label(self, index):
        """
        Load labels by index
        :param index:
        :return: label in bool
        """
        self.logger.debug("Loading label {}".format(index))

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

    def save_model(self, model, name='detection_model', compress=3):
        """
        Save model to model directory
        :param model:
        :param name:
        :param compress: compression level, 3 is recommended
        :return:
        """
        model_pickle = os.path.join(self.model_dir, name + '.pkl')
        joblib.dump(model, model_pickle, compress=compress)

        self.logger.info('{0} saved to {1}'.format(name, model_pickle))

    def save_all(self, image_pipeline, feature_pipeline, image_classifier):
        detection_model = DetectionModel(img_ppl=image_pipeline,
                                         ftr_ppl=feature_pipeline,
                                         img_clf=image_classifier)
        self.save_model(detection_model, name='detection_model')

    def load_model(self, name='detection_model'):
        """
        Load model from model directory
        :param name:
        :return:
        """
        model_pkl = os.path.join(self.model_dir, name + '.pkl')
        model = joblib.load(model_pkl)

        self.logger.info('{0} load from {1}'.format(name, model_pkl))
        return model

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
