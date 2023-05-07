import os
import sys

from PyQt5.QtWidgets import QApplication
from data.loaders.CifarDataLoader import CifarDataLoader
from data.normalizators.DivideNormalizator import DivideNormalizator
from gui.Application import Application
from models.comparators.ModelComparator import ModelComparator

from models.creators.CifarModelsCreator import CifarModelsCreator
from models.creators.DogsModelsCreator import DogsModelsCreator
from models.loaders.TensorModelLoader import TensorModelLoader
from models.savers.TensorModelSaver import TensorModelSaver

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def workbench():
    normalizator = DivideNormalizator(255.0)
    data_loader = CifarDataLoader()
    train_data = data_loader.load_train_data()
    train_data = normalizator.normalize(train_data)
    test_data = data_loader.load_test_data()
    test_data = normalizator.normalize(test_data)

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    models_creator = CifarModelsCreator()

    cnn_model = models_creator.create_advanced_neural_model()
    cnn_second_model = models_creator.create_advanced_neural_model()
    simple_model = models_creator.create_simple_neural_model()
    path = os.path.normpath('C:/Users/Kerwus/Downloads/fat.jpg')
    simple = data_loader.load_single_image(path)
    simple = normalizator.normalize(simple)

    batch_size = 64
    epochs = 1
    verbose = 1
    cnn_model.train(train_data, epochs=epochs, number_of_samples=batch_size, verbose=verbose)
    cnn_second_model.train(train_data, epochs=epochs, number_of_samples=batch_size, verbose=verbose)
    simple_model.train(train_data, epochs=epochs, number_of_samples=batch_size, verbose=verbose)
    idx = cnn_model.predict(simple)[0]
    print(class_names[idx])
    comparator = ModelComparator(simple)
    print("comparator")
    print(comparator.predict_compare(cnn_model, simple_model))
    print(comparator.predict_compare(cnn_model, cnn_model))
    print(comparator.predict_compare(cnn_model, cnn_second_model))

def init_app(data_loader, normalizator):
    loader = TensorModelLoader()
    cnn_model = loader.load_model("./cifarCnn.h5")
    text_file = open("cifarClasses.txt", "r")
    class_names = text_file.read().split(',')
    print(class_names)
    app = QApplication(sys.argv)
    my_app = Application(data_loader, cnn_model, normalizator, class_names)
    sys.exit(app.exec_())

def train_model(data_loader, normalizator):
    models_creator = DogsModelsCreator()
    cnn_model = models_creator.create_simple_neural_model(10)
    saver = TensorModelSaver()
    train_data = data_loader.load_train_data()
    train_data = normalizator.normalize(train_data)
    batch_size = 64
    epochs = 1
    verbose = 1
    cnn_model.train(train_data, epochs=epochs, number_of_samples=batch_size, verbose=verbose)
    saver.save_model(cnn_model, os.path.normpath("./dogsCifarCnn.h5"))

data_loader = CifarDataLoader()
normalizator = DivideNormalizator(255.0)
train_model(data_loader, normalizator)
init_app(data_loader, normalizator)
#TODO : dogs data loader




