#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
import os
from keras import initializers



# In[2]:


im = cv2.imread('phone_dataset/1/frame0.jpg')
phone_directory = "phone_dataset/"
smoking_directory = "smoking_dataset/"


# In[3]:


print(type(im))
im.shape


# #### Poblado de training set

# In[14]:


print('Loading data...')
#video_set = np.empty((74,1024,1280,3))
n_videoseq = len(os.listdir(phone_directory))
video_set_small = np.empty((n_videoseq,30,240,320,3))
video_set = np.empty((n_videoseq,30,480,640,3))
for sub_d in range(n_videoseq):
    for img_ite in range(len(os.listdir(phone_directory+str(sub_d)))):
        video_set[sub_d,img_ite,:,:,:] = cv2.resize(cv2.imread('phone_dataset/{}/frame{}.jpg'.format(sub_d,img_ite)),(640,480))
        video_set_small[sub_d,img_ite,:,:,:] = cv2.resize(cv2.imread('phone_dataset/{}/frame{}.jpg'.format(sub_d,img_ite)),(320,240))
        video_set[sub_d,img_ite,:,:,:]       /=255
        video_set_small[sub_d,img_ite,:,:,:] /=255
        
print('Load complete')


# In[15]:


print('Video shape:')
print(video_set.shape)
print(video_set_small.shape)


# In[16]:


plt.imshow(video_set[1,20,:,:,2])

y_train = pd.read_csv('train_labels.csv')
y_train = y_train.drop('id',axis=1)
# In[13]:


#print(video_set_small[1,20,:,:,2])


# In[17]:


list = os.listdir('phone_dataset')
n_f=len(list)
print(n_f)


# In[ ]:


print('Creating model..')
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv3D, MaxPooling3D
from keras.optimizers import SGD, RMSprop
#input_shape = video_set_small.shape


# In[ ]:


def create_model():
    model = Sequential()
    #model.add(Conv3D(4, kernel_size=(3, 3, 1), activation='relu', padding='same', input_shape=(30,480,640,3)))
    model.add(Conv3D(4, kernel_size=(3, 3, 1), activation='relu', padding='same', input_shape=(30,240,320,3),kernel_initializer=initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)))
    model.add(Conv3D(8, kernel_size=(3, 3, 1), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 1)))

    model.add(Dropout(0.25))
    model.add(Conv3D(16, kernel_size=(3, 3, 1), activation='relu'))
    model.add(Conv3D(32, kernel_size=(3, 3, 1), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 1)))
    model.add(Dropout(0.25))
    model.add(Conv3D(64, kernel_size=(3, 3, 1), activation='relu'))

    model.add(MaxPooling3D(pool_size=(2, 2, 1)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


# In[ ]:


model = create_model()
model.summary()
print('Model complete')


# In[ ]:

rms = RMSprop(lr=0.0001, rho=0.95, epsilon=1e-6)
model.compile(optimizer='RMSprop',loss='binary_crossentropy',metrics=['accuracy'])


# In[ ]:


model.fit(video_set_small,y_train,batch_size = 2, epochs=20)


# In[ ]:




