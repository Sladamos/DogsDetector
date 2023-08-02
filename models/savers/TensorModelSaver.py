from models.savers.ModelSaver import ModelSaver


class TensorModelSaver(ModelSaver):
    def save_model_with_classes(self, model, classes, saver_config):
        model_path = saver_config["model_path"]
        classes_path = saver_config["classes_path"]
        model.model.save(model_path)
        with open(classes_path, "w") as f:
            [f.write(line + '\n') for line in classes]
