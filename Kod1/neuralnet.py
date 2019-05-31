import numpy as np
import time
from datetime import timedelta
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger
import gzip
import sys
import pickle
import pandas
from keras.preprocessing import image
import cv2
from keras.datasets import mnist

start_time = time.monotonic()

f = gzip.open('mnist.pkl.gz', 'rb')
if sys.version_info < (3,):
    data = pickle.load(f)
else:
    data = pickle.load(f, encoding='bytes')
f.close()

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
(X_train, y_train), (X_test, y_test) = data

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

n_classes = 10

Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)

model = Sequential()
model.add(Dense(512, input_dim=784))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

history = model.fit(X_train, Y_train,
          batch_size=64, epochs=10,
          verbose=2,
          validation_data=(X_test, Y_test))

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))

import numpy
loss_history = history.history["loss"]
acc_history = history.history["acc"]
numpy_loss_history = numpy.array(loss_history)
numpy_acc_history = numpy.array(acc_history)
numpy.savetxt("loss_history.txt", numpy_loss_history,delimiter=",")
numpy.savetxt("acc_history.txt", numpy_acc_history,delimiter=",")

import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model.save('neuralnet.h5')
#tesztrész
'''
image_index = 334
plt.imshow(X_test[image_index].reshape(28, 28))
pred = model.predict(X_test[image_index].reshape(1, 784))
print(pred.argmax())
plt.show()
'''

