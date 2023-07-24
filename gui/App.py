from abc import abstractmethod


class App:

    def __init__(self, data_loader, detector):
        self.data = None
        self.detector = detector
        self.target_size = (224, 224)
        self.data_loader = data_loader
        self.call_simple_model()
        self.show()

    @abstractmethod
    def select_image(self):
        pass

    @abstractmethod
    def show(self):
        pass

    def get_classification_result(self):
        if self.data is None:
            return "Incorrect image"
        else:
            return self.detector.classify(self.data)

    def set_data(self, data):
        self.data = data

    def call_simple_model(self):
        self.detector.select_mode("simple")

    def call_transfered_model(self):
        self.detector.select_mode("transfered")
