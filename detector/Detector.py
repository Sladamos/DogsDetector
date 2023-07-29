import os


class Detector:

    def __init__(self, model_loader, models_configs):
        self.target_size = None
        self.model = None
        self.class_names = None
        self.model_loader = model_loader
        self.models_configs = models_configs

    def classify(self, image):
        result = self.model.predict(image)[0]
        return self.class_names[result]

    def select_mode(self, mode):
        if mode in self.models_configs:
            model_path = self.models_configs[mode]["model_path"]
            self.model = self.model_loader.load_model(model_path)
            classes_path = self.models_configs[mode]["classes_path"]
            self.class_names = self.get_classes_from(classes_path)

    def get_classes_from(self, classes_path):
        path = os.path.normpath(classes_path)
        with open(path) as f:
            lines = f.read().splitlines()
        return lines
