import os
import sys

from PyQt5.QtWidgets import QApplication
from keras.preprocessing.image import ImageDataGenerator

from data.loaders.CifarDataLoader import CifarDataLoader
from data.loaders.DogsDataLoader import DogsDataLoader
from data.normalizators.DivideNormalizator import DivideNormalizator
from gui.Application import Application
from models.comparators.ModelComparator import ModelComparator


import tensorflow as tf
import matplotlib.pyplot as plt
from models.creators.CifarModelsCreator import CifarModelsCreator
from models.creators.DogsModelsCreator import DogsModelsCreator
from models.loaders.TensorModelLoader import TensorModelLoader
from models.savers.TensorModelSaver import TensorModelSaver

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def make_plots(results, name):
    plt.plot(results.history['accuracy'])
    plt.plot(results.history['val_accuracy'])
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['training', 'validation'], loc='upper left')
    plt.savefig('./newHope/accuracy/' + name)
    plt.show()

    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training', 'validation'], loc='upper left')
    plt.savefig('./newHope/loss/' + name)
    plt.show()


def workbench():

    batch_size = 64
    epochs = 10
    verbose = 1
    image_size = 224
    dataset_path = os.path.normpath("./images/dogs/Images")
    training_set = datagen.flow_from_directory(dataset_path, target_size=(image_size, image_size),
                                               batch_size=batch_size, class_mode='categorical', subset='training')

    validation_set = datagen.flow_from_directory(dataset_path, target_size=(image_size, image_size),
                                                 batch_size=batch_size, class_mode='categorical', subset='validation')
    input_shape = (224, 224, 3)

    dirs = os.listdir("./images/dogs/Images")
    class_names = [dir.split('-', 1)[1] for dir in dirs]
    models_creator = DogsModelsCreator(len(class_names))

    # loader = TensorModelLoader()
    # cnn_model = loader.load_model("./newHope/simple.h5")
    cnn_model = models_creator.create_simple_neural_model(input_shape)
    results = cnn_model.model.fit(x=training_set, validation_data=validation_set, epochs=epochs, batch_size=batch_size,
                                  verbose=verbose)
    make_plots(results, "loss_simple.png")
    saver = TensorModelSaver()
    saver.save_model(cnn_model, os.path.normpath("./newHope/simple.h5"))


def init_app():
    data_loader = CifarDataLoader()
    normalizator = DivideNormalizator(255.0)
    loader = TensorModelLoader()
    cnn_model = loader.load_model("./advancedSecndCnn.h5")
    dirs = os.listdir("./images/dogs/Images")
    class_names = [dir.split('-', 1)[1] for dir in dirs]

    #class_names = text_file.read().split(',')
    print(class_names)
    return data_loader, cnn_model, normalizator, class_names


def train_model():
    batch_size = 64
    dirs = os.listdir("./images/dogs/Images")
    models_creator = DogsModelsCreator(len(dirs))
    data_loader = DogsDataLoader(batch_size)
    cnn_model = models_creator.create_simple_neural_model(input_shape=(224, 224, 3))
    loader = TensorModelLoader()

    saver = TensorModelSaver()
    train_data = data_loader.load_train_data()
    validation_data = data_loader.load_validation_data()
    epochs = 15
    verbose = 1
    results = cnn_model.train_with_validation(train_data, validation_data, epochs=epochs, batch_size=batch_size, verbose=verbose)

    make_plots(results, "model_8.png")
    #saver.save_model(cnn_model, os.path.normpath("./newHope/model.h5"))


# workbench()
train_model()
# data_loader, cnn_model, normalizator, class_names = init_app()
# app = QApplication(sys.argv)
# my_app = Application(data_loader, cnn_model, normalizator, class_names)
# sys.exit(app.exec_())
