import tensorflow as tf
from tensorflow import keras

from __init__ import jena_full_pipeline
from preprocessing import WindowGenerator

train, dev, test = jena_full_pipeline()

