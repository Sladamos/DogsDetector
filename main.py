import os

from models.comparators.ModelComparator import ModelComparator

from models.creators.models.CifarModelsCreator import CifarModelsCreator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras

import matplotlib.pyplot as plt


cifar10 = keras.datasets.cifar10

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
models_creator = CifarModelsCreator()

cnn_model = models_creator.create_convolution_neural_model()
simple_model = models_creator.create_simple_neural_model()
# training
batch_size = 64
epochs = 1
cnn_model.train(train_images, train_labels, epochs=epochs, number_of_samples=batch_size, verbose=1)
simple_model.train(train_images, train_labels, epochs=epochs, number_of_samples=batch_size, verbose=1)

comparator = ModelComparator(test_images, test_labels)
print("comparator")
print(comparator.compare(cnn_model, simple_model, batch_size))
print(comparator.compare(simple_model, cnn_model, batch_size))
print(comparator.compare(simple_model, simple_model, batch_size))