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

## Demo
The service is provided with a simple demo ```demo.py```, illustrating the main interaction.
The demo, as the service, has to be run in the same environment.
The demo source is located in the folder ```scripts```.

## Compose timeseries from xml files
Compose timeseries from .zip archive(XMLs):
```
python compose_timeseries.py <zipPath>
```
The zip must contain the series, organized in groups of files xml.
The script is located in the folder ```scripts```.

## Tests
The test suite developed for the service is available under the folder ```tests```.
The suite is developed accordingly to the framework pytest.
The environment contains all the dependencies to run the test suite.
To run the suite once the environment is activated:
```
pytest
```