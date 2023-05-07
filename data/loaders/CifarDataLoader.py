import numpy as np
import cv2

from data.Data import Data
from data.loaders.DataLoader import DataLoader
from tensorflow import keras
from keras.preprocessing.image import image_utils


class CifarDataLoader(DataLoader):
    def __init__(self):
        self.cifar10 = keras.datasets.cifar10

    def load_train_data(self):
        (train_images, train_labels), (test_images, test_labels) = self.cifar10.load_data()
        size = len(train_images)
        data = []
        while size > 0:
            img = train_images[size-1]
            train_images = train_images[:-1]
            large_img = cv2.resize(img, dsize=(227, 227), interpolation=cv2.INTER_CUBIC)
            data.append(large_img)
            size -= 1
            if size == 40000:
                break
        data = np.array(data)
        data = Data(data, train_labels)
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