#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install kaggle')


# In[12]:


get_ipython().system('pip install opencv-python')


# In[15]:


get_ipython().system(' pip install -q kaggle')


# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import tensorflow as tf

from tqdm.auto import tqdm

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, MaxPool2D, Activation
from keras.utils import to_categorical
from IPython.display import SVG
from tensorflow.keras.utils import image_dataset_from_directory 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img 
from tensorflow.keras.preprocessing import image_dataset_from_directory 

import os
import cv2
import shutil
import matplotlib.image as mpimg


# In[8]:


get_ipython().system(' mkdir -p ~/.kaggle')
get_ipython().system(' cp kaggle.json ~/.kaggle/')


# In[9]:


get_ipython().system('kaggle datasets download - d salader/dogs-vs-cats')


# In[5]:


from zipfile import ZipFile 
  
data_path = "C:/Users/asus/Downloads/Dogs-vs-Cats.zip"
  
with ZipFile(data_path, 'r') as zip: 
    zip.extractall("C:/Users/asus/Downloads/Dogs-vs-Cats") 
    print('The data set has been extracted.') 


# # Data Preperation for TRaining 

# In[6]:


base_dir = "C:/Users/asus/Downloads/Dogs-vs-Cats"
  
# Create datasets 
train_datagen = image_dataset_from_directory(base_dir, 
                                                  image_size=(256,256), 
                                                  subset='training', 
                                                  seed = 1, 
                                                 validation_split=0.1, 
                                                  batch_size= 32) 
test_datagen = image_dataset_from_directory(base_dir, 
                                                  image_size=(256,256), 
                                                  subset='validation', 
                                                  seed = 1, 
                                                 validation_split=0.1, 
                                                  batch_size= 32)


# In[7]:


## Normalizing 

def process(image, label):
    image= tf.cast(image/255, tf.float32)
    return image, label

train_datagen = train_datagen.map(process)
test_datagen = test_datagen.map(process)


# # Model Building

# In[8]:


model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),padding="valid", activation='relu', input_shape=(256,256,3)))
model.add(MaxPool2D(pool_size=(2,2), strides=2, padding='valid'))

model.add(Conv2D(64,kernel_size=(3,3),padding="valid", activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=2, padding='valid'))

model.add(Conv2D(128,kernel_size=(3,3),padding="valid", activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=2, padding='valid'))

model.add(Flatten())

model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[9]:


model.summary()


# # Model Compilation

# In[11]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# # Model Training

# In[ ]:


history = model.fit(train_datagen, epochs=10, validation_data = test_datagen)


# In[ ]:


plt.plot(history.history['accuracy'], color='red', label='train')
plt.plot(history.history['val_accuracy'], color='red', label='validation')
plt.legend()
plt.show()


# In[ ]:


plt.plot(history.history['loss'], color='red', label='train')
plt.plot(history.history['val_loss'], color='red', label='validation')
plt.legend()
plt.show()


# # Model Testing and Prediction

# In[ ]:


from keras.preprocessing import image 
  
#Input image 
test_image = image.load_img("C:/Users/asus/Downloads/cat image.avif",target_size=(256,256)) 
  
#For show image 
plt.imshow(test_image) 
test_image = image.img_to_array(test_image) 
test_image = np.expand_dims(test_image,axis=0) 
  
# Result array 
result = model.predict(test_image) 
  
#Mapping result array with the main name list 
i=0
if(result>=0.5): 
    print("Dog") 
else: 
    print("Cat")


# In[ ]:





# In[ ]:




