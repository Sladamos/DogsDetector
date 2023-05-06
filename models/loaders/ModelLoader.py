from abc import abstractmethod


class ModelLoader:
    @abstractmethod
    def load_model(self, path):
        pass
