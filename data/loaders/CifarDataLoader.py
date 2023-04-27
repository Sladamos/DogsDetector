from data.Data import Data
from data.loaders.DataLoader import DataLoader
from tensorflow import keras

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

    def load_single_data(self):
        #TODO implement if prediction implemented
        pass