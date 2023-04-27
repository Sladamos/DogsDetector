from abc import abstractmethod


class Normalizator:
    @abstractmethod
    def normalize(self, image):
        pass