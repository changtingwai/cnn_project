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

#deep learning 训练层
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(3,width,height)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000))
model.add(Activation('softmax'))

print "train begin"
WEIGHTS_FNAME = 'vgg16_weights.h5'
model.load_weights(WEIGHTS_FNAME)
print type(model)
new_model =Sequential()
new_model.add(ZeroPadding2D((1,1),input_shape=(3,width,height)))
new_model.add(Convolution2D(64, 3, 3, activation='relu'))
new_model.add(ZeroPadding2D((1,1)))
new_model.add(Convolution2D(64, 3, 3, activation='relu'))
new_model.add(MaxPooling2D((2,2), strides=(2,2)))

new_model.add(ZeroPadding2D((1,1)))
new_model.add(Convolution2D(128, 3, 3, activation='relu'))
new_model.add(ZeroPadding2D((1,1)))
new_model.add(Convolution2D(128, 3, 3, activation='relu'))
new_model.add(MaxPooling2D((2,2), strides=(2,2)))

new_model.add(ZeroPadding2D((1,1)))
new_model.add(Convolution2D(256, 3, 3, activation='relu'))
new_model.add(ZeroPadding2D((1,1)))
new_model.add(Convolution2D(256, 3, 3, activation='relu'))
new_model.add(ZeroPadding2D((1,1)))
new_model.add(Convolution2D(256, 3, 3, activation='relu'))
new_model.add(MaxPooling2D((2,2), strides=(2,2)))

new_model.add(ZeroPadding2D((1,1)))
new_model.add(Convolution2D(512, 3, 3, activation='relu'))
new_model.add(ZeroPadding2D((1,1)))
new_model.add(Convolution2D(512, 3, 3, activation='relu'))
new_model.add(ZeroPadding2D((1,1)))
new_model.add(Convolution2D(512, 3, 3, activation='relu'))
new_model.add(MaxPooling2D((2,2), strides=(2,2)))

new_model.add(ZeroPadding2D((1,1)))
new_model.add(Convolution2D(512, 3, 3, activation='relu'))
new_model.add(ZeroPadding2D((1,1)))
new_model.add(Convolution2D(512, 3, 3, activation='relu'))
new_model.add(ZeroPadding2D((1,1)))
new_model.add(Convolution2D(512, 3, 3, activation='relu'))
new_model.add(MaxPooling2D((2,2), strides=(2,2)))

narray = model.get_weights()[0:26]
new_model.set_weights(narray)

#构造数据
#X = 6000
#x1 = 30000
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
        if k%100 == 0:
            print "加载 %d 数据" %k
    #分割训练集,测试集
    length = 25000
    X_train = X[:length,:]
    X_test = X[length+1:,:]

    y = np_utils.to_categorical(y, 4)
    y_train = y[:length,:]
    y_test = y[length+1:,:]
print X_train.shape
print X_test.shape
print y_train.shape
print y_test.shape
np.save('X1_train.npy',X_train)
np.save('y1_train.npy',y_train)
np.save('X1_test.npy',X_test)
np.save('y1_test.npy',y_test)
print "数据准备完成"

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model_weights_path = 'own_weights.h5'
print "全连接训练"
#自己的全连接
# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=new_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(4, activation='softmax'))

if os.path.exists(top_model_weights_path):
    top_model.load_weights(top_model_weights_path)
else:
    X_top_train = new_model.predict(X_train,batch_size=32)
    print "全连接网络搭建完毕"
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    top_model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])
    top_model.fit(X_top_train, y_train, batch_size=30, nb_epoch=10,shuffle=True,verbose=1)
    top_model.save_weights(top_model_weights_path)
print "全连接训练结束"
# add the model on top of the convolutional base
new_model.add(top_model)

#冻结
for layer in new_model.layers[:20]:
    layer.trainable = False


WEIGHTS_FNAME = 'picture_years_vgg16.h5'
sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
new_model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])
new_model.fit(X_train, y_train, batch_size=30, nb_epoch=50,shuffle=True,verbose=1,validation_data=(X_test,y_test))
new_model.save_weights(WEIGHTS_FNAME)
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