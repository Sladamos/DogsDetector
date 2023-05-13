from models.savers.ModelSaver import ModelSaver

import tensorflow as tf

class TensorModelSaver(ModelSaver):
    def save_model(self, model, path):
        model.model.save(path)
