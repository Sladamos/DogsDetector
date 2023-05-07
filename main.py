import os
import sys

from PyQt5.QtWidgets import QApplication
from keras.preprocessing.image import ImageDataGenerator

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
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.3)

    batch_size = 64
    epochs = 5
    verbose = 1
    image_size = 224
    dataset_path = os.path.normpath("./images/dogs/Images")
    training_set = datagen.flow_from_directory(dataset_path, target_size=(image_size, image_size),
                                               batch_size=batch_size, class_mode='categorical', subset='training')

    validation_set = datagen.flow_from_directory(dataset_path, target_size=(image_size, image_size),
                                                 batch_size=batch_size, class_mode='categorical', subset='validation')

    models_creator = DogsModelsCreator(120)
    cnn_model = models_creator.create_advanced_neural_model()
    cnn_model.model.fit(x=training_set, validation_data=validation_set, epochs=epochs, batch_size=batch_size, verbose=verbose)


def init_app():
    data_loader = CifarDataLoader()
    normalizator = DivideNormalizator(255.0)
    loader = TensorModelLoader()
    cnn_model = loader.load_model("./cifarCnn.h5")
    text_file = open("cifarClasses.txt", "r")
    class_names = text_file.read().split(',')
    print(class_names)
    return data_loader, cnn_model, normalizator, class_names

def train_model():
    data_loader = CifarDataLoader()
    normalizator = DivideNormalizator(255.0)
    models_creator = CifarModelsCreator()
    cnn_model = models_creator.create_advanced_neural_model()
    saver = TensorModelSaver()
    train_data = data_loader.load_train_data()
    train_data = normalizator.normalize(train_data)
    batch_size = 64
    epochs = 5
    verbose = 1
    cnn_model.train(train_data, epochs=epochs, number_of_samples=batch_size, verbose=verbose)
    saver.save_model(cnn_model, os.path.normpath("./cifarCnn.h5"))

workbench()
#train_model()
#data_loader, cnn_model, normalizator, class_names = init_app()
#TODO : dogs model creator and dogs data loader

#app = QApplication(sys.argv)
#my_app = Application(data_loader, cnn_model, normalizator, class_names)
#sys.exit(app.exec_())


