from abc import abstractmethod


class ModelSaver:
    @abstractmethod
    def save_model(self, model, path):
        pass