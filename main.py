import os

from data.loaders.CifarDataLoader import CifarDataLoader
from data.normalizators.DivideNormalizator import DivideNormalizator
from models.comparators.ModelComparator import ModelComparator

from models.creators.models.CifarModelsCreator import CifarModelsCreator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

normalizator = DivideNormalizator(255.0)
data_loader = CifarDataLoader()
train_data = data_loader.load_train_data()
train_data = normalizator.normalize(train_data)
train_images, train_labels = train_data.get_images(), train_data.get_labels()
test_data = data_loader.load_test_data()
test_data = normalizator.normalize(test_data)
test_images, test_labels = test_data.get_images(), test_data.get_labels()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
models_creator = CifarModelsCreator()

cnn_model = models_creator.create_convolution_neural_model()
simple_model = models_creator.create_simple_neural_model()
# training
batch_size = 64
epochs = 1
cnn_model.train(train_images, train_labels, epochs=epochs, number_of_samples=batch_size, verbose=0)
simple_model.train(train_images, train_labels, epochs=epochs, number_of_samples=batch_size, verbose=0)

comparator = ModelComparator(test_images, test_labels)
print("comparator")
print(comparator.evaluate_compare(cnn_model, simple_model, batch_size))
print(comparator.evaluate_compare(simple_model, cnn_model, batch_size))
print(comparator.evaluate_compare(simple_model, simple_model, batch_size))