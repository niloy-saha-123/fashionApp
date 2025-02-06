import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt


# Load the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize the images to the range 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Expand dimensions to add the channel (grayscale image has 1 channel)
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=10, validation_split=0.1)


test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")


predictions = model.predict(x_test)

# Plot the first image and its prediction
plt.imshow(x_test[0].squeeze(), cmap='gray')
plt.title(f"Predicted label: {tf.argmax(predictions[0])}")
plt.show()
