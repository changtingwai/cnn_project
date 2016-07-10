#coding:UTF-8
import scipy.io as scio
from keras.utils import np_utils
import os
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D,ZeroPadding2D
from keras.optimizers import SGD
from keras.layers import BatchNormalization
width = 128
height = 128
def ImageToMatrix(filename):
    # 读取图片
    im = Image.open(filename).convert('RGB')
    im = im.resize((width,height))
    arr = np.asarray(im,dtype='float32')/255.0
    return arr

File_name = '/Users/baoshiqi/job/wiki_crop/'
dataFile = File_name+'wiki.mat'
data = scio.loadmat(dataFile)
photo_path = data['wiki']['full_path'][0][0][0]
photo_size = data['wiki']['face_location'][0][0][0]


model = Sequential()
model.add(Convolution2D(32,3,3,border_mode='valid',input_shape=(3,width,height),init='uniform'))
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
#构造数据
#num=62328
print "开始准备数据"
count = 0
k = 0
picture_num = 30000
X = np.empty((picture_num,3,width,height),dtype='float32')
y = np.empty((picture_num,),dtype='uint8')
import os
if os.path.exists('X1_test.npy'):
    X_train = np.load('X1_train.npy')
    y_train = np.load('y1_train.npy')
    X_test = np.load('X1_test.npy')
    y_test = np.load('y1_test.npy')
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
        X[k,:,:,:] = [new_Matrix[:,:,0],new_Matrix[:,:,1],new_Matrix[:,:,2]]
        y[k] = person_age
        k = k + 1
    #分割训练集,测试集
    length = 25000
    X = np.asarray(X)
    X_train = X[:length,:]
    X_test = X[length+1:,:]

    y = np_utils.to_categorical(y, 4)
    y_train = y[:length,:]
    y_test = y[length+1:,:]
np.save('X1_train.npy',X_train)
np.save('y1_train.npy',y_train)
np.save('X1_test.npy',X_test)
np.save('y1_test.npy',y_test)
print X_train.shape
print X_test.shape
print y_train.shape
print y_test.shape
print "数据准备完成"

WEIGHTS_FNAME = 'picture_years_vgg16.h5'
model.load_weights(WEIGHTS_FNAME)

sgd = SGD(lr=0.1, decay=1e-8, momentum=0.9, nesterov=True)
from keras.optimizers import RMSprop
rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
# model.compile(loss='mean_squared_error', optimizer=rms,metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])
print "开始训练"
WEIGHTS_FNAME_next = 'picture_years_vgg16_2.h5'
model.fit(X_train, y_train, batch_size=500, nb_epoch=100,shuffle=True,verbose=1,validation_data=(X_test,y_test))
model.save_weights(WEIGHTS_FNAME_next)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])1
print('Test accuracy:', score[1])
y_pred = model.predict(X_test)
from sklearn.metrics import classification_report
print y_pred.shape
print y_test.shape
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test,axis=1)
print classification_report(y_test,y_pred)

