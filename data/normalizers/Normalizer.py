from abc import abstractmethod


class Normalizer:
    @abstractmethod
    def normalize(self, image):
        pass