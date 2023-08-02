import os
from abc import ABC, abstractmethod

from displayer.Plotter import Plotter
from initializers.trainer.TensorflowTrainerInitializer import TensorflowTrainerInitializer
from option.Option import Option


class TrainOption(Option, ABC):
    def __init__(self):
        self.initializers = {
            "tensorflow": TensorflowTrainerInitializer
        }

    @abstractmethod
    def create_raw_model(self, models_creator, input_shape):
        pass

    @abstractmethod
    def get_config_name(self):
        pass

    def execute(self, config):
        app_config_name = self.get_config_name()
        app_config = config[app_config_name]
        data_loader, data_normalizer, model_operator, model_saver = self.initialize_trainer(app_config)
        #TO TRAINER
        model = self.get_model(app_config, model_operator)
        classes = self.get_classes(app_config["training_set_path"])
        results = self.execute_train(model, app_config, data_loader)
        # TO TRAINER
        model_saver.save_model_with_classes(model, classes, app_config["save"])
        self.make_plots(results, app_config["plots"])
        # TODO: deal with checkpoints and save classes to file

    def execute_train(self, model, app_config, data_loader):
        training_features = self.get_training_features(app_config)
        batch_size = training_features["batch_size"]
        epochs = training_features["epochs"]
        verbose = training_features["verbose"]
        train_data, validation_data = self.get_datas(app_config, data_loader)
        results = model.train_with_validation(train_data, validation_data, epochs=epochs, batch_size=batch_size,
                                              verbose=verbose, callbacks=[])
        return results

    def get_training_features(self, app_config):
        trainer_type = app_config["trainer_type"]
        return app_config["training_features"][trainer_type]

    def load_model_from_path(self, model_loader, model_path):
        return model_loader.load_model(model_path)

    def initialize_trainer(self, app_config):
        initializer_type = app_config["trainer_type"]
        initializer_config = self.create_initializer_config(app_config)
        initializer = self.initializers[initializer_type]()
        return initializer.initialize_trainer(initializer_config)

    def calculate_number_of_classes(self, app_config):
        path = os.path.normpath(app_config["training_set_path"])
        dirs = os.listdir(path)
        return len(dirs)

    def create_initializer_config(self, app_config):
        number_of_classes = self.calculate_number_of_classes(app_config)
        initializer_config = {
            "train_from_existing": app_config["train_from_existing_model"],
            "number_of_classes": number_of_classes,
            "detected_type": app_config["detected_type"]
        }
        return initializer_config

    def make_plots(self, results, plotter_config):
        plotter = Plotter(plotter_config)
        plotter.print_plots(results)

    def get_datas(self, app_config, data_loader):
        train_data = data_loader.load_train_data(app_config["training_set_path"])
        validation_data = data_loader.load_validation_data(app_config["validation_set_path"])
        return train_data, validation_data

    def get_model(self, app_config, model_operator):
        training_features = self.get_training_features(app_config)
        input_shape = tuple(training_features["input_shape"])
        if app_config["train_from_existing_model"]:
            model = self.load_model_from_path(model_operator, app_config["existing_model_path"])
        else:
            model = self.create_raw_model(model_operator, input_shape)
        return model

    def get_classes(self, images_path):
        dirs = sorted(os.listdir(images_path))
        return [dir.split('-', 1)[1].replace("_", " ").capitalize() for dir in dirs]
