from configparser import ConfigParser

conf = ConfigParser()
conf.read('config.ini')

def download_dataset():
    dset = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip'
    print(f'Download: {dset}')
    import os
    import datetime

    os.chdir('data/datasets')

    import tensorflow as tf

    zip_path = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
        fname='jena_climate_2009_2016.csv.zip',
        extract=True)
    csv_path, _ = os.path.splitext(zip_path)
    print(csv_path)


if __name__ == "__main__":
    download_dataset()