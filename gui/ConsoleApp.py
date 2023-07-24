import os

from gui.App import App


class ConsoleApp(App):

    def __init__(self, data_loader, model_loader):
        super().__init__(data_loader, model_loader)
        self.is_program_launched = false
        options = {
            "a": b,
        }

    def show(self):
        self.is_program_launched = true
        while self.is_program_launched:
            #select option
            #parse option


    def select_image(self):
        pass
