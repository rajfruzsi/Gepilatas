from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import matplotlib
from keras.datasets import mnist
import matplotlib.pyplot as plt

model=load_model('./neuralnet.h5')
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.load_weights('./neuralnet.h5')
(X_train, y_train), (X_test, y_test) = mnist.load_data()

image_index = 334
plt.imshow(X_test[image_index].reshape(28, 28))
pred = model.predict(X_test[image_index].reshape(1, 784))
print(pred.argmax())
plt.show()
