from abc import abstractmethod


class App:

    def __init__(self, data_loader=None, data_normalizer=None, detector=None, init_model=None):
        if detector is None:
            return
        self.image = None
        self.detector = detector
        self.data_loader = data_loader
        self.data_normalizer = data_normalizer
        self.detector_modes = {
            'simple': self.select_simple_model,
            'transferred': self.select_transferred_model
        }
        self.detector_modes[init_model]()

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
        self.image = self.data_normalizer.normalize(self.image)

    def forget_image(self):
        self.image = None

    def is_image_selected(self):
        return self.image is not None

    def get_detector_modes(self):
        return self.detector_modes

    def select_simple_model(self):
        self.detector.select_mode("simple")

    def select_transferred_model(self):
        self.detector.select_mode("transferred")
