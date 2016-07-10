#coding:UTF-8
import scipy.io as scio
from keras.utils import np_utils
import os
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

def ImageToMatrix(filename):
    # 读取图片
    width = 128
    height = 128

    im = Image.open(filename)
    im = im.resize((width,height))
    if im.mode == 'RGB':
        # im.show()
        data = im.getdata()
        data = np.array(data,dtype='float')/255.0
        new_data = data.reshape(3,width,height)
        return new_data
#     new_im = Image.fromarray(new_data)
#     # 显示图片
#     new_im.show()
def MatrixToImage(data):
    data = data*255
    new_data = data.reshape(16384,3)
    new_im = Image.frombuffer(mode='rgb',size=49152,data=data)
    return new_im

File_name = '/Users/baoshiqi/job/wiki_crop/'
dataFile = File_name+'wiki.mat'
data = scio.loadmat(dataFile)
photo_path = data['wiki']['full_path'][0][0][0]
photo_size = data['wiki']['face_location'][0][0][0]
picture_num = 7328
X = np.empty((picture_num,3,128,128))
y = []
#构造数据
#num=62328
print "开始准备数据"
for i in range(0,7328):
    image_path = photo_path[i][0]
    file = File_name +image_path
    #将矩阵装进X里
    new_Matrix = ImageToMatrix(file)
    X[i] = new_Matrix
    # X.append(new_Matrix)
    age_array = photo_path[i][0].split('_')
    birth_date = age_array[1]
    birth_year = birth_date.split('-')[0]
    take_date = age_array[2]
    take_year = take_date.split('.')[0]
    person_age = int(take_year) - int(birth_year)
    if person_age<20:
        person_age = 0
    elif person_age<30:
        person_age = 1
    elif person_age<50:
        person_age = 2
    else:
        person_age = 3
    y.append(person_age)
#list 转换矩阵
length = 7000
X = np.asarray(X)
X_train = X[:length,:]
X_test = X[length+1:,:]
print X_train.shape
print X_test.shape
y = np_utils.to_categorical(y, 4)
y_train = y[:length,:]
y_test = y[length+1:,:]
print y_train.shape
print y_test.shape

print "数据准备完成"
#deep learning 训练层

model = Sequential()
model.add(Convolution2D(32,3,3,border_mode='valid',input_shape=(3,128,128)))
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

model.add(Dense(4))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])
print "train begin"
WEIGHTS_FNAME = 'mnist_cnn_weights.hdf'
if os.path.exists(WEIGHTS_FNAME):
    print "loading exists weights not train again"
    model.load_weights(WEIGHTS_FNAME)
else:
    model.fit(X_train, y_train, batch_size=100, nb_epoch=2,shuffle=True,verbose=1,validation_data=(X_test,y_test))
    model.save_weights(WEIGHTS_FNAME)
model.fit(X_train, y_train, batch_size=100, nb_epoch=2,shuffle=True,verbose=1,validation_data=(X_test,y_test))
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