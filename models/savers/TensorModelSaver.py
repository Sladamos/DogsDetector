from models.savers.ModelSaver import ModelSaver


class TensorModelSaver(ModelSaver):
    def save_model(self, model, path):
        model.model.save(path)
