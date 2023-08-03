import os
from abc import abstractmethod


class Trainer:

    @abstractmethod
    def create_raw_model(self, models_creator, input_shape):
        pass

    def train_with_config(self, trainer_config, model_operator, data_loader, callbacks):
        model = self.get_model(trainer_config, model_operator)
        classes = self.get_classes(trainer_config["training_set_path"])
        results = self.execute_train(model, trainer_config, data_loader, callbacks)
        return model, classes, results

    def execute_train(self, model, trainer_config, data_loader, callbacks):
        training_features = self.get_training_features(trainer_config)
        batch_size = training_features["batch_size"]
        epochs = training_features["epochs"]
        verbose = training_features["verbose"]
        train_data, validation_data = self.get_datas(trainer_config, data_loader)
        results = model.train_with_validation(train_data, validation_data, epochs=epochs, batch_size=batch_size,
                                              verbose=verbose, callbacks=callbacks)
        return results

    def get_datas(self, trainer_config, data_loader):
        train_data = data_loader.load_train_data(trainer_config["training_set_path"])
        validation_data = data_loader.load_validation_data(trainer_config["validation_set_path"])
        return train_data, validation_data

    def get_model(self, trainer_config, model_operator):
        training_features = self.get_training_features(trainer_config)
        input_shape = tuple(training_features["input_shape"])
        if trainer_config["train_from_existing_model"]:
            model = self.load_model_from_path(model_operator, trainer_config["existing_model_path"])
        else:
            model = self.create_raw_model(model_operator, input_shape)
        return model

    def get_classes(self, images_path):
        dirs = sorted(os.listdir(images_path))
        return [dir.split('-', 1)[1].replace("_", " ").capitalize() for dir in dirs]

    def get_training_features(self, trainer_config):
        trainer_type = trainer_config["trainer_type"]
        return trainer_config["training_features"][trainer_type]

    def load_model_from_path(self, model_loader, model_path):
        return model_loader.load_model(model_path)
