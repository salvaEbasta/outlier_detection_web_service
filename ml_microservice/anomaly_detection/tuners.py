class Tuner():
    def tune(self, X, y=None):
        """
        Param:
        - X : timeseries, already preprocessed, array like
        - y : objective values
        """
        raise NotImplementedError()

class AbstractTuner(Tuner):
    def __init__(self):
        self._search_space = {}
        self.best_model = None
        self.best_config = None
        self.results = []

    @property
    def search_space(self):
        return self._search_space

    def save_results(self, path):
        pass

    
class WindGaussTuner(AbstractTuner):
    def __init__(self):
        super().__init__()
