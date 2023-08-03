from option.train.TrainOption import TrainOption
from trainers.SimpleTrainer import SimpleTrainer


class TrainSimpleOption(TrainOption):

    def execute(self, config):
        app_config = config["simple"]
        trainer = SimpleTrainer()
        self.execute_training(app_config, trainer)
