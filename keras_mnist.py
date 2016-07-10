#coding:UTF-8
import input_data
import os
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
def ImageToMatrix(filename):
    # 读取图片
    im = Image.open(filename)
    # 显示图片
#     im.show()
    width,height = im.size
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data,dtype='float')/255.0
    new_data = np.reshape(data,(width,height))
    return new_data
#     new_im = Image.fromarray(new_data)
#     # 显示图片
#     new_im.show()
def MatrixToImage(data):
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
X = mnist.train.images
y = mnist.train.labels
X_test = mnist.test.images
y_test = mnist.test.labels
X_train = X.reshape(X.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
print type(X_train)
print X_train.shape
exit(0)
# i = 46
# new_im = MatrixToImage(X_train[i,0])
# new_im.show()
# print("label : ", y[i,:])
nb_classes = 10

model = Sequential()
model.add(Convolution2D(32,3,3,border_mode='valid',input_shape=(1,28,28)))
# model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, 256, 256)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
# keras会自动计算输入shape
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])

WEIGHTS_FNAME = 'mnist_cnn_weights.hdf'
if os.path.exists(WEIGHTS_FNAME):
    print "loading exists weights not train again"
    model.load_weights(WEIGHTS_FNAME)
else:
    model.fit(X_train, y, batch_size=100, nb_epoch=2,shuffle=True,verbose=1,validation_data=(X_test,y_test))
    model.save_weights(WEIGHTS_FNAME)
model.fit(X_train, y, batch_size=100, nb_epoch=2,shuffle=True,verbose=1,validation_data=(X_test,y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
y_pred = model.predict(X_test)
from sklearn.metrics import classification_report
print y_pred.shape
print y_test.shape
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test,axis=1)
print classification_report(y_test,y_pred)