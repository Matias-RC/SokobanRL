
class Replayer:

    def __init__(self, data_generator, trainer):

        self.data_generator = data_generator
        self.trainer = trainer

    def do(self, session, model):

        print("Replayer: do")
        return model