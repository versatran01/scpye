import os
import re

import cv2
import joblib
from scpye.improc.image_processing import u8_from_bw


class ImageDataset(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_dir = os.path.join(self.data_dir, 'image')
        self.train_dir = os.path.join(self.data_dir, 'train')
        self.model_dir = os.path.join(self.data_dir, 'model')
        self.detect_dir = os.path.join(self.data_dir, 'detect')
        self.track_dir = os.path.join(self.data_dir, 'track')

        self.image_name = 'image_rect_color'
        self.label_name = 'image_rect_label'
        self.image_ext = 'png'
        self.image_fmt = '{0}_{1:05d}.{2}'

    def _save_image(self, image_dir, index, prefix, image):
        image_name = self.image_fmt.format(prefix, index, self.image_ext)
        image_name_full = os.path.join(image_dir, image_name)
        cv2.imwrite(image_name_full, image)

    def _load_image(self, image_dir, index, prefix, color=True):
        image_name = self.image_fmt.format(prefix, index, self.image_ext)
        image_name_full = os.path.join(image_dir, image_name)
        flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE

        image = cv2.imread(image_name_full, flag)

        if image is None:
            raise ValueError('image not found. {}'.format(image_name_full))

        return image

    def load_image(self, index):
        return self._load_image(self.image_dir, index, self.image_name)

    def load_train_image(self, index):
        return self._load_image(self.train_dir, index, self.image_name)

    def load_label(self, index):
        label = self._load_image(self.train_dir, index, self.label_name,
                                 color=False)
        return u8_from_bw(label, val=1)

    def load_train(self):
        index_list = []
        regex = re.compile(r'\d+')
        file_list = os.listdir(self.train_dir)
        for file_name in sorted(file_list):
            if self.image_name in file_name and file_name.endswith(
                    self.image_ext):
                index_list.append(int(regex.search(file_name).group(0)))
        if len(index_list) == 0:
            raise ValueError('Empty index list')

        Is, Ls = [], []
        for index in index_list:
            Is.append(self.load_train_image(index))
            Ls.append(self.load_label(index))

        return Is, Ls

    def save_model(self, fd, model_name='model.pkl'):
        """
        :param fd: FruitDetector
        :param model_name:
        :return:
        """
        model_name_full = os.path.join(self.model_dir, model_name)
        joblib.dump(fd, model_name_full)

    def load_model(self, model_name='model.pkl'):
        """
        :param model_name:
        :return:
        """
        model_name_full = os.path.join(self.model_dir, model_name)
        return joblib.load(model_name_full)

    def save_detect(self, index, bgr, bw):
        """
        :param index:
        :param bgr:
        :param bw:
        :return:
        """
        self._save_image(self.detect_dir, index, 'detect_color', bgr)
        self._save_image(self.detect_dir, index, 'detect_label', bw)
