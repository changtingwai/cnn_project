#coding:UTF-8
import input_data
import os
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
import scipy.io as scio
from keras.utils import np_utils
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

#构造数据
#num=62328
print "开始准备数据"
count = 0
k = 0
picture_num = 60000
X = np.empty((picture_num/10,3,128,128))
y = []
import os
if os.path.exists('X_test.npy'):
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
else:
    for i in range(0,picture_num):
        count = count + 1
        image_path = photo_path[i][0]
        file = File_name +image_path
        #获得图片矩阵
        new_Matrix = ImageToMatrix(file)

        #获得图片年龄
        age_array = photo_path[i][0].split('_')
        birth_date = age_array[1]
        birth_year = birth_date.split('-')[0]
        take_date = age_array[2]
        take_year = take_date.split('.')[0]
        person_age = int(take_year) - int(birth_year)
        if person_age<25:
            person_age = 0
        elif person_age<33:
            person_age = 1
        elif person_age<48:
            person_age = 2
        else:
            person_age = 3
        if count % 10 == 0:
            X[k] = new_Matrix
            y.append(person_age)
            k = k + 1
    #分割训练集,测试集
    length = 5500
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
    # np.save('X_train.npy',X_train)
    # np.save('y_train.npy',y_train)
    np.save('X_test.npy',X_test)
    # np.save('y_test.npy',y_test)

print "数据准备完成"


model = Sequential()
model.add(Convolution2D(32,3,3,border_mode='valid',input_shape=(3,128,128),init='uniform'))
# model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, 256, 256)))
model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
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

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])
print "开始训练网络"
model.fit(X_train, y_train, batch_size=10, nb_epoch=200,shuffle=True,verbose=1,validation_data=(X_test,y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])