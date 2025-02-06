import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize the data (scale pixel values to range [0, 1])
x_train = x_train / 255.0
x_test = x_test / 255.0
