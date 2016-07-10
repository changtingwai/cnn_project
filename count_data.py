#coding:UTF-8
import scipy.io as scio
from PIL import Image
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt

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
picture_num = 30000
X = np.empty((picture_num,3,128,128))
y = []
count = 0
#构造数据
for i in range(0,picture_num):
    count = count + 1
    image_path = photo_path[i][0]
    file = File_name +image_path
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

    y.append(person_age)
#list 转换矩阵
plt.hist(y,bins=4)
plt.show()
X = np.asarray(X)
X_train = X[:60000,:]
X_test = X[60001:,:]

y = np_utils.to_categorical(y, 4)
Y_train = y[:60000,:]
Y_test = y[60001:,:]
