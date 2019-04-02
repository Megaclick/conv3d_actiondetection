import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv3D, MaxPooling3D
from keras.optimizers import SGD
from keras.optimizers import RMSprop, Adam, SGD, Adadelta
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from keras import initializers

print('Setting initializers parameters..')

path = 'phone_dataset/'
labels = pd.read_csv('train_labels.csv')


def get_img(path,height,width,norm = False):
    video_set = np.empty((30,width,height,3))
    count = 0
    for img_in_dir in os.listdir(path):
        img_path = path+img_in_dir
        img = cv2.imread(img_path)
        if norm:
            video_set[count,:,:,:] = (cv2.resize(img,(height,width))/255).astype(np.float16)
        else:
            video_set[count,:,:,:] = cv2.resize(img,(height,width)).astype(np.int16)

        count+=1
    return video_set

def get_label(id_,labels):
    lab = labels.loc[int(id_)]['label']
    return lab

def tensor_generator(path,files,label_file,batch_size=1):
    while 1:
        batch_paths = np.random.choice(a = files ,size=batch_size)
        batch_input=[]
        batch_output=[]
        for input_path in batch_paths:
            input_ = get_img(path+input_path+'/',640,480,norm=True)
            output_ = get_label(input_path,label_file)
            batch_input+=[input_]
            batch_output+=[output_]
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)
        yield (batch_x,batch_y)
print('Load complete..')

print('Creating model..')

def create_model():
    model = Sequential()
    model.add(Conv3D(8, kernel_size=(3, 3, 3), activation='relu', padding='same', input_shape=(30,480,640,3)))
    #model.add(Conv3D(4, kernel_size=(3, 3, 1), activation='relu', padding='same', input_shape=(30,240,320,3),kernel_initializer=initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(16, kernel_size=(3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu'))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))


    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='softmax'))

    return model


model = create_model()
model.summary()
print('Model complete')
model.compile(optimizer=Adadelta(lr=0.1),loss='binary_crossentropy',metrics=['accuracy'])
model.fit_generator(tensor_generator(path,os.listdir('phone_dataset/'),labels),
 epochs = 35, steps_per_epoch=32,use_multiprocessing=True)

model.save('weights.h5')
model.save_weights('asdasd.h5')
print('Model Saved..')