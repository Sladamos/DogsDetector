import os


class Detector:

    def __init__(self, model_loader, models_paths, images_paths):
        self.target_size = None
        self.model = None
        self.class_names = None
        self.model_loader = model_loader
        self.models_paths = models_paths
        self.images_paths = images_paths

    def classify(self, image):
        result = self.model.predict(image)[0]
        return self.class_names[result]

    def select_mode(self, mode):
        if mode in self.models_paths:
            model_path = self.models_paths[mode]
            self.model = self.model_loader.load_model(model_path)
            images_path = self.images_paths[mode]
            dirs = sorted(os.listdir(images_path))
            self.class_names = [directory.split('-', 1)[1].replace("_", " ").capitalize() for directory in dirs]
