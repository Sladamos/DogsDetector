import os

class App:

    def __init__(self, data_loader, model_loader):
        self.class_names = None
        self.data = None
        self.model = None
        self.target_size = (224, 224)
        self.data_loader = data_loader
        self.model_loader = model_loader
        self.set_label("")
        self.call_simple_model()
        self.show()

    @abstractmethod
    def select_image(self):
        pass

    @abstractmethod
    def show(self):
        pass

    def identify_image(self):
        if self.data is None:
            return

        data = self.data
        result = self.model.predict(data)[0]
        self.set_label(self.class_names[result])

    def set_data(self, data):
        self.data = data

    def call_simple_model(self):
        self.model = self.model_loader.load_model("./newHope/simple")
        dirs = sorted(os.listdir("./images/dogs/Images"))
        self.class_names = [dir.split('-', 1)[1].replace("_", " ").capitalize() for dir in dirs]

    def call_transfered_model(self):
        self.model = self.model_loader.load_model("./newHope/transfered")
        dirs = sorted(os.listdir("./images/dogs/All_images"))
        self.class_names = [dir.split('-', 1)[1].replace("_", " ").capitalize() for dir in dirs]
