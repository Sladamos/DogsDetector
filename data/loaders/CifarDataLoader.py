import numpy as np

from data.Data import Data
from data.loaders.DataLoader import DataLoader
from tensorflow import keras
from keras.preprocessing.image import image_utils


class CifarDataLoader(DataLoader):
    def __init__(self):
        self.cifar10 = keras.datasets.cifar10

    def load_train_data(self):
        (train_images, train_labels), (test_images, test_labels) = self.cifar10.load_data()
        data = Data(train_images, train_labels)
        return data

    def load_test_data(self):
        (train_images, train_labels), (test_images, test_labels) = self.cifar10.load_data()
        data = Data(test_images, test_labels)
        return data

    def load_single_image(self, img_path, target_size=(32, 32)):
        img = image_utils.load_img(img_path, target_size=target_size, color_mode="rgb")
        img = image_utils.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        data = Data(img, None)
        return data