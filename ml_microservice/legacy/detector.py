class Detector():
    def __init__(self, window_size=50, l=0.01, k=2, forecasting_model="linear", path=None):
        """ 
            Specify path to load a Detector from a folder. The other parameters will be ignored even if specified.\n
            l: lambda\n
            k: multiplicator for variance to compose the threshold\n
            window_size: the dimension of the input\n
            Assume the label to be monodimensional\n
        """
        if path is None:
            self._window_size = window_size

            self._forecasting_model = forecasting_model
            self._forecaster_factory = model_factory.ForecasterFactory(window_size)
            if not self._forecaster_factory.has(forecasting_model):
                raise ValueError(
                    f'The forecasting model selected, \'{forecasting_model}\', is not supported')
            self._hyper_forecaster = self._forecaster_factory.build(self._forecasting_model)
            
            self._best_forecaster = None
            self._best_epoch = -1
            self._best_params = None

            self._lambda = l
            self._k = k
            self._var = 0
            self._mean = 0

            self._detection_history = History()
        else:
            self._load(path)

    @property
    def window_size(self):
        return self._window_size

    @property
    def params(self) -> Dict:
        """
            A dict with all the parameters
        """
        return dict(
            l=self._lambda,
            window_size=self.window_size,
            k=self._k,
            variance=self._var,
            mean=self._mean,
            forecasting_model=self._forecasting_model,
            best_epoch=self._best_epoch,
            best_hyperparams=self._best_params,
        )

    @property
    def history(self):
        return self._detection_history

    def fit(self, X, y, dev_data: Tuple, max_epochs=50, patience=5):
        """
            -> predictor_history, best_params, epochs, detection_history: History
        """
        tuner = kt.Hyperband(
            self._hyper_forecaster,
            objective='val_loss',
            max_epochs=max_epochs,
            factor=3,
            # project_name=f"{time.time()}",
        )
        early_call = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
        )
        tuner.search(X, y,
                    epochs=max_epochs,
                    validation_data=dev_data,
                    callbacks=[early_call]
                    )
        best_hps = tuner.get_best_hyperparameters()[0]
        #print(f"Best hyperparams: {best_hps.values}")
        self._best_params = best_hps.values

        # Find best epoch
        self._best_forecaster = tuner.hypermodel.build(best_hps)
        history = self._best_forecaster.fit(X, y,
                                            validation_data=dev_data,
                                            epochs=max_epochs,
                                            callbacks=[early_call]
                                            )

        val_loss_history = history.history['val_loss']
        self._best_epoch = val_loss_history.index(min(val_loss_history)) + 1

        # Train best model + best epoch
        self._best_forecaster = tuner.hypermodel.build(best_hps)
        
        history = self._best_forecaster.fit(X, y,
                                            validation_data=dev_data,
                                            epochs=self._best_epoch,
                                            )
        
        # train the threshold
        self.update_variance(*dev_data)
        return history, best_hps.values, self._best_epoch, self._detection_history

    def update_variance(self, X, y):
        assert len(y.shape) == 2 and y.shape[1] == 1

        y_hat = self._best_forecaster.predict(X)
        y_naive = metrics.naive_prediction(X)
        assert len(y_hat.shape) == 2 and y_hat.shape[1] == 1
        
        self._detection_history.update_state(y, y_hat, y_naive)
        rmse = self._detection_history.rmse
        self._mean = np.mean(rmse)
        self._var = np.var(rmse)

    @property
    def threshold(self):
        return self._mean + self._k * np.sqrt(self._var)

    def detect(self, X, y):
        """-> np.array(N x 1), 0: expected, 1: anomaly; prediction; detection_history"""
        assert len(y.shape) == 2 and y.shape[1] == 1

        y_hat = self._best_forecaster.predict(X)
        assert len(y_hat.shape) == 2 and y_hat.shape[1] == 1
        anomalies = np.abs(y_hat - y) > self.threshold

        y_naive = metrics.naive_prediction(X)
        history = History()
        history.update_state(y, y_hat, y_naive)
        return anomalies.astype(int), y_hat, history

    def detect_update(self, X, y):
        """
            -> anomalies; y_hat; detection_history: History\n
            Sequences in X and y respect the overall temporal order in which they are extracted
            1: anomaly, 0: expected
        """
        assert len(y.shape) == 2 and y.shape[1] == 1

        y_hat = self.predict(X)
        assert len(y_hat.shape) == 2 and y_hat.shape[1] == 1
        y_naive = metrics.naive_prediction(X)
        anomalies = []
        for i in range(y_hat.shape[0]):
            rmse = self._detection_history.rmse
            self._mean = np.mean(rmse)
            self._var = np.var(rmse)
            anomalies.append([np.abs(y_hat[i] - y[i]) > self.threshold])
            self._detection_history.update_state(y[i], y_hat[i], y_naive[i])
        return np.array(anomalies).astype(int), y_hat, self._detection_history

    def predict(self, X):
        return self._best_forecaster.predict(X)

    def save(self, ddir, param_file = "params.json"):
        if not os.path.exists(ddir):
            os.makedirs(ddir)
        if self._best_forecaster is not None:
            self._best_forecaster.save(ddir)
        
        self._detection_history.save(ddir)

        param_file = os.path.join(ddir, param_file)
        with open(param_file, 'w') as f:
            json.dump(self.params, f)
        

    def _load(self, env, param_file = 'params.json', history_file = 'history.csv'):
        with open(os.path.join(env, 'params.json'), 'r') as f:
            params = json.load(f)
        self._window_size = int(params['window_size'])
        self._lambda = float(params['l'])
        self._k = float(params['k'])
        self._var = np.float(params['variance'])
        self._mean = np.float(params['mean'])

        self._forecasting_model = params['forecasting_model']
        self._forecaster_factory = model_factory.ForecasterFactory(self.window_size)
        self._hyper_forecaster = self._forecaster_factory.build(self._forecasting_model)
        self._best_forecaster = keras.models.load_model(env)
        self._best_epoch = int(params['best_epoch'])
        self._best_params = params['best_hyperparams']

        self._detection_history = History()
        self._detection_history.load(os.path.join(env, 'history.csv'))


class History():
    def __init__(self):
        self._h = pd.DataFrame({
            'timestamp': [],
            'datapoints': [],
            'mse': [],
            'mse_naive': [],
            'y': [],
            'y_hat': [],
            'y_naive': []
        })

    def update_state(self, y, y_hat, y_naive):
        y = np.array(y).flatten()
        y_hat = np.array(y_hat).flatten()
        y_naive = np.array(y_naive).flatten()

        if len(self._h) > 0:
            mse_old = self._h['mse'].to_numpy()[-1]
            dpoints_old = self._h['datapoints'].to_numpy()[-1]
            mse_naive_old = self._h['mse_naive'].to_numpy()[-1]
        else:
            mse_old = np.finfo(float).eps
            dpoints_old = np.finfo(float).eps
            mse_naive_old = np.finfo(float).eps
        
        for i, ys in enumerate(zip(y, y_hat, y_naive)):
            dpoints = i + 1 + dpoints_old
            mse_old = (ys[0] - ys[1])**2/dpoints + mse_old*dpoints_old/dpoints
            mse_naive_old = (ys[0] - ys[2])**2/dpoints + mse_naive_old*dpoints_old/dpoints
            self._h = self._h.append(pd.DataFrame({
                                        'timestamp': [datetime.datetime.now()],
                                        'datapoints': [dpoints],
                                        'mse': [mse_old],
                                        'mse_naive': [mse_naive_old],
                                        'y': [ys[0]],
                                        'y_hat': [ys[1]],
                                        'y_naive': [ys[2]],
                                    }), ignore_index=True)

    def merge(self, another):
        self.update_state(
            another._h['y'], 
            another._h['y_hat'],
            another._h['y_naive']
        )

    @property
    def values(self):
        return self._h.to_dict()

    @property
    def rmse(self):
        if len(self._h) > 0:
            return np.sqrt(self._h['mse'].to_numpy())
        else:
            return np.array([])
    
    @property
    def naive_score(self):
        if len(self._h) > 0:
            return np.divide(self._h['mse'].to_numpy(), self._h['mse_naive'].to_numpy())
        else:
            return np.array([])

    def save(self, ddir, history_file = "history.csv"):
        f = os.path.join(ddir, history_file)
        if os.path.exists(f):
            os.remove(f)
        self._h.to_csv(f)

    def load(self, path):
        if not os.path.exists(path):
            logging.warning(f"History load: no path {path}")
            self.__init__()
        else:
            self._h = pd.read_csv(path)
            if 'Unnamed: 0' in self._h:
                self._h = self._h.drop('Unnamed: 0', axis = 1)
