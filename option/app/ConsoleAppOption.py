from gui.ConsoleApp import ConsoleApp
from option.app.AppOption import AppOption


class ConsoleAppOption(AppOption):
    def execute(self, config):
        app_config = config["console"]
        init_config = config["init"]
        data_loader, data_normalizer, detector = self.initialize_app(init_config, app_config)
        app = ConsoleApp(data_loader, data_normalizer, detector, app_config)
        app.show()
