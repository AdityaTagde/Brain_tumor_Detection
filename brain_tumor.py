#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf 
from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential,Model 
from tensorflow import keras 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

import PIL
import os 


# In[2]:


path=os.path.join('C:\\Users\\ASUS TUF F15\\Projects\\Brain_tumor\\Brain_tumor_dataset')


# In[3]:


path


# In[4]:


os.listdir(path)


# In[5]:


os.listdir('C:\\Users\\ASUS TUF F15\\Projects\\Brain_tumor\\Brain_tumor_dataset\\Training' )


# In[6]:


os.listdir('C:\\Users\\ASUS TUF F15\\Projects\\Brain_tumor\\Brain_tumor_dataset\\Training\\glioma')


# In[7]:


PIL.Image.open('C:\\Users\\ASUS TUF F15\\Projects\\Brain_tumor\\Brain_tumor_dataset\\Training\\glioma\\Tr-gl_0015.jpg')


# In[8]:


train_dir=os.path.join(path,'Training')


# In[9]:


test_dir=os.path.join(path,'Testing')


# In[10]:


train_dir


# In[11]:


train_ds=tf.keras.utils.image_dataset_from_directory(train_dir,image_size=(224,224))


# In[12]:


test_ds=tf.keras.utils.image_dataset_from_directory(test_dir,image_size=(224,224))


# In[13]:


class_n=train_ds.class_names


# In[14]:


class_n


# # Showing Samples

# In[15]:


plt.figure(figsize=(10,10))
for images,labels in train_ds.take(1): 
    for i in range(9): 
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('int'))
        plt.title(class_n[labels[i]])
        plt.axis('off')


# In[16]:


def image_preprocess(image,label): 
    image=keras.applications.resnet50.preprocess_input(image) 
    label=tf.cast(label,tf.float32)
    return image,label


# In[17]:


autotune=tf.data.AUTOTUNE


# In[18]:


train_ds=train_ds.map(image_preprocess).prefetch(buffer_size=autotune)
test_ds=test_ds.map(image_preprocess).prefetch(buffer_size=autotune) 


# # Loading pretrained model 

# In[19]:


base_model=tf.keras.applications.ResNet50(input_shape=(224,224,3),include_top=False,weights='imagenet',pooling='avg')


# In[20]:


base_model.summary()


# In[21]:


base_model.trainable=False


# In[22]:


X=layers.Dense(22,'relu')(base_model.output)
X=layers.Dropout(0.3)(X) 

out=layers.Dense(4,'softmax')(X)


# In[23]:


model=Model(inputs=base_model.input,outputs=out)


# In[24]:


model.summary()


# In[25]:


model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[26]:


history=model.fit(train_ds,validation_data=test_ds,epochs=16)


# # Evaluation

# In[27]:


loss,acc=model.evaluate(test_ds)


# In[29]:


plt.figure(figsize=(14, 5))
# plot for accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], color='green', linestyle='-', marker='o', markersize=5, label='Train')
plt.plot(history.history['val_accuracy'], color='red', linestyle='--', marker='x', markersize=5, label='Val')

plt.title('Model Accuracy Over Epochs', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left', fontsize=12)
plt.ylim(0, 1)

# plot for loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], color='green', linestyle='-', marker='o', markersize=5, label='Train')
plt.plot(history.history['val_loss'], color='red', linestyle='--', marker='x', markersize=5, label='Val')

plt.title('Model Loss Over Epochs', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left', fontsize=12)
plt.xlim(0, len(history.history['loss']))

plt.tight_layout()
plt.show()


# # Testing the model 

# In[30]:


from keras.utils import load_img,img_to_array


# In[41]:


test_img=load_img('meningioma.jpg',target_size=(224,224))


# In[42]:


test_img


# In[43]:


img_ar=img_to_array(test_img).reshape(1,224,224,3)


# In[44]:


img_ar


# In[45]:


img_ar.shape


# In[46]:


class_n[np.argmax(model.predict(img_ar))]


# # Saving the model 

# In[47]:


model.save('brain_tumor.h5')


# In[ ]:




