# Machine Learning Microservice for Outlier Detection in Timeseries
A small http web service to detect outliers in a given timeseries.
Use state-of-art methods like:
 - **Windowed Gaussian**
 - **Prophet** + **Windowed Gaussian**
 - **SARIMAX** + **Windowed Gaussian**
 - **DeepAnT** + **Empirical Rule**
 - **Stacked LSTM** + **Empirical Rule**
 - **Stacked GRU** + **Empirical Rule**

## Service launch
The service needs a particular environmento to run properly. The env is described in the conda-env.yml file.
The file describes a Conda environment.
To setup the environment, create a new conda environment and activate it:
```
conda env create -f conda-env.yml
conda activate tesi-env
```
To launch the service, once the environment is active, simply run:
```
python ml_microservice
```

Launch service: python -m ml_microservice

Demo: python long_demo.ipynb

Compose timeseries from .zip archive(XMLs): python compose_timeseries.py zipPath
