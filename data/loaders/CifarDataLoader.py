from data.loaders.DataLoader import DataLoader
from tensorflow import keras

class CifarDataLoader(DataLoader):
    def __init__(self):
        self.cifar10 = keras.datasets.cifar10

    def load_train_images(self):
        (train_images, train_labels), (test_images, test_labels) = self.cifar10.load_data()
        return train_images, train_labels

    def load_test_images(self):
        (train_images, train_labels), (test_images, test_labels) = self.cifar10.load_data()
        return test_images, test_labels

    def load_image(self):
        #TODO implement with prediction