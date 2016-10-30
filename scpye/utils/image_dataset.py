import os
import logging
import re

import cv2
import numpy as np
from sklearn.externals import joblib

from scpye.detect.fruit_detector import FruitDetector
from scpye.improc.image_processing import u8_from_bw
from scpye.utils.exception import ImageNotFoundError


class ImageDataset(object):
    def __init__(self, data_dir, image_name, image_ext='png'):
        self.data_dir = data_dir
        self.image_name = image_name
        self.image_ext = image_ext

    def _load_image(self, index, prefix, color=True):
        file_fmt = '{}_{}.png'
        file_name = file_fmt.format(prefix, index)
        file_name_full = os.path.join(self.data_dir, file_name)
        flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE

        image = cv2.imread(file_name_full, flag)

        if image is None:
            raise ImageNotFoundError(file_name_full)

        return image


class TrainingSet(ImageDataset):
    def __init__(self, data_dir, image_name, label_name):
        super(TrainingSet, self).__init__(data_dir, image_name)
        self.label_name = label_name
        self.index_list = []

        regex = re.compile(r'\d+')
        file_list = os.listdir(self.data_dir)
        for file_name in sorted(file_list):
            if self.image_name in file_name and file_name.endswith(
                    self.image_ext):
                self.index_list.append(regex.search(file_name).group(0))

    def load_image(self, index):
        return self._load_image(index, self.image_name)

    def load_label(self, index):
        return self._load_image(index, self.label_name, color=False)

    def load_image_label_list(self):
        if len(self.index_list) == 0:
            raise ValueError('Empty index list')

        Is, Ls = [], []
        for index in self.index_list:
            Is.append(self.load_image(index))
            Ls.append(u8_from_bw(self.load_label(index), val=1))
        return Is, Ls


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
        return u8_from_bw(label, val=1)

    def load_image_and_label(self, index):
        """
        :param index: index of image
        :return: image and label
        """
        image = self.load_image(index)
        label = self.load_label(index)
        return image, label

    def save_detector(self, model, name='fruit_detector', compress=3):
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
        fruit_detector = FruitDetector(img_ppl=image_pipeline,
                                       ftr_ppl=feature_pipeline,
                                       img_clf=image_classifier)
        self.save_detector(fruit_detector, name='fruit_detector')

    def load_detector(self, name='fruit_detector'):
        """
        Load detector from model directory
        :param name:
        :return:
        """
        detector_pickle = os.path.join(self.model_dir, name + '.pkl')
        fruit_detector = joblib.load(detector_pickle)

        self.logger.info('{0} load from {1}'.format(name, detector_pickle))
        return fruit_detector

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
