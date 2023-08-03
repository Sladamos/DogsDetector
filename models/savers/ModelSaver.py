from abc import abstractmethod


class ModelSaver:
    @abstractmethod
    def save_model_with_classes(self, model, classes, saver_config):
        pass
