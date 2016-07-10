#coding:utf-8
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Convolution2D,Flatten,MaxPooling2D
from keras.optimizers import SGD
model = Sequential()

import numpy as np
data = np.random.random((1000,100,100))
print data
labels = np.random.randint(2,size=(1000,1))

model.add(Convolution2D(32,3,3,border_mode='valid',input_shape=(1,1000,10000)))
# model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, 256, 256)))
model.add(Activation('relu'))
# model.add(Convolution2D(32, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Convolution2D(64, 3, 3, border_mode='valid'))
# model.add(Activation('relu'))
# model.add(Convolution2D(64, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
# keras会自动计算输入shape
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(data, labels, batch_size=100, nb_epoch=1,shuffle=True,verbose=1,validation_split=0.2)
