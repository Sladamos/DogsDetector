import os

from data.Data import Data
from data.loaders.DataLoader import DataLoader
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DogsDataLoader(DataLoader):

    def load_validation_data(self):
        validation_set = self.datagen.flow_from_directory(self.dataset_path, target_size=self.target_size,
                                                          batch_size=self.batch_size, class_mode='categorical',
                                                          subset='validation')

        data = Data(validation_set, None)
        return data

    def __init__(self, batch_size):
        self.datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.05,
            rotation_range=5,
            zoom_range=0.25,
            horizontal_flip=True,
            validation_split=0.2)
        self.dataset_path = os.path.normpath("./images/dogs/Images")
        self.batch_size = batch_size
        self.target_size = (224, 224)

    def load_train_data(self):
        training_set = self.datagen.flow_from_directory(self.dataset_path, target_size=self.target_size,
                                                        batch_size=self.batch_size, class_mode='categorical',
                                                        subset='training')

        data = Data(training_set, None)
        return data

    def load_test_data(self):
        pass

    def load_single_image(self, img_path, target_size):
        pass
