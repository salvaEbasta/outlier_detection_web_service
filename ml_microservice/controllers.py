class Controller():
    """
        Template design pattern
    """
    def __init__(self):
        pass

    def handle(self):
        return self.run()

    def run(self):
        raise NotImplementedError('Not implemented')