from abc import abstractmethod


class App:

    def __init__(self, data_loader, data_normalizator, detector):
        self.image = None
        self.detector = detector
        self.data_loader = data_loader
        self.data_normalizator = data_normalizator
        #self.select_simple_model()

    @abstractmethod
    def select_image_path(self):
        pass

    @abstractmethod
    def show(self):
        pass

    def get_classification_result(self):
        if self.image is None:
            return "Incorrect image"
        else:
            return self.detector.classify(self.image)

    def load_image(self, image_path):
        self.image = self.data_loader.load_single_image(image_path)
        self.image = self.data_normalizator.normalize(self.image)

    def select_simple_model(self):
        self.detector.select_mode("simple")

    def select_transfered_model(self):
        self.detector.select_mode("transfered")
