import os

import numpy as np
from keras.utils import image_utils

from data.Data import Data
from data.loaders.DataLoader import DataLoader
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DogsDataLoader(DataLoader):

    def load_validation_data(self):
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=0.2)
        validation_set = datagen.flow_from_directory(self.dataset_path, target_size=self.target_size,
                                                     batch_size=self.batch_size, class_mode='categorical',
                                                     subset='validation')

        data = Data(validation_set, None)
        return data

    def __init__(self, batch_size):
        self.dataset_path = os.path.normpath("./images/dogs/Images")
        self.batch_size = batch_size
        self.target_size = (224, 224)

    def load_train_data(self):
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.05,
            rotation_range=10,
            zoom_range=0.25,
            horizontal_flip=True,
            validation_split=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2)
        training_set = datagen.flow_from_directory(self.dataset_path, target_size=self.target_size,
                                                   batch_size=self.batch_size, class_mode='categorical',
                                                   subset='training')

        data = Data(training_set, None)
        return data

    def load_test_data(self):
        pass

    def load_single_image(self, img_path, target_size=(224, 224)):
        img = image_utils.load_img(img_path, target_size=target_size, color_mode="rgb")
        img = image_utils.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        data = Data(img, None)
        return data
