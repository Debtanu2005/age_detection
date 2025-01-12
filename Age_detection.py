#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# In[4]:


df = pd.read_csv(r"C:\Users\aspir\Downloads\UTKFace.csv")
df


# In[ ]:


df['pixels'][0][0]


# In[ ]:


import cv2


# In[9]:


def pixel_to_image(arr):
    arr= arr.split()
    arr= np.array(arr, dtype=np.float32)
    arr= arr.reshape((24,32,3))
    return arr


# In[10]:


df['images']= df['pixels'].apply(pixel_to_image)


# In[11]:


#def resize_image(img)
#    img= img/255.
#    resized_img = cv2.resize(img, (256, 256))
#    return resized_img


# In[ ]:





# In[12]:


df['images']= df['images']/255.


# In[13]:


df['age'].value_counts()


# In[14]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler


# In[15]:


std= StandardScaler()


# In[16]:


age_2d = np.array(df['age']).reshape(-1, 1)


# In[17]:


df['age']= std.fit_transform(age_2d)


# In[18]:


arr= np.array(df['age'])


# In[19]:


df['age']


# In[20]:


import seaborn as sns


# In[21]:


df1= df[['age','ethnicity']]


# In[22]:


data= df1.corr()


# In[23]:


sns.heatmap(data)


# In[24]:


df2=df[['age','images']]


# In[25]:


df


# In[26]:


def resize_image(img):
    
    resized_img = np.array(tf.image.resize(img, (128, 128)))
    return resized_img


# In[27]:


df['images']= df['images'].apply(resize_image)


# In[28]:


plt.imshow(df['images'][2500])


# In[29]:


from sklearn.model_selection import train_test_split
x= np.array(df['images'])
y= np.array(df['age'])


# In[30]:


x_train, x_test, y_train,y_test= train_test_split(x,y, random_state=48, test_size=0.2)


# In[31]:


print(x_train.shape)
print(y_train.shape)
x_test.shape[0]


# In[32]:


x_train =np.stack(x_train)


# In[33]:


x_test= np.stack(x_test)


# In[34]:


y_test


# In[ ]:





# In[37]:


from tensorflow.keras.layers import Dense , GlobalAveragePooling2D, Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.models import Model


# In[38]:


input_shape= Input(shape=(128,128,3))

base_model= EfficientNetB4(include_top=False,
                          weights= 'imagenet')
output= base_model(input_shape)

avg_pool= GlobalAveragePooling2D()(output)

Out= Dense(1,activation='linear')(avg_pool)

model= Model(inputs= input_shape, outputs= Out)

model.summary()


# In[33]:


model.compile(
    loss=tf.keras.losses.MeanAbsoluteError(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['mae']  
)


# In[42]:


model.fit(x_train, y_train,
         epochs=5,
          validation_data=(x_test,y_test),
          batch_size=32,
          validation_steps=x_test.shape[0]
         )


# In[39]:


img_input = Input(shape=(128,128,3))

x = Conv2D(128, (3, 3), activation='relu')(img_input)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (2, 2), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x= Conv2D(128, (3,3))(x)
x = Flatten()(x)
output = Dense(1, activation='linear')(x) 
model2 = Model(inputs=img_input, outputs=output)


# In[40]:


model2.compile(
    loss=tf.keras.losses.MeanAbsoluteError(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['mae']  
)


# In[ ]:





# In[ ]:


history = model2.fit(x_train, y_train,
         epochs=10,
          validation_data=(x_test,y_test), batch_size=32)


# In[37]:


loss= pd.DataFrame(history.history)


# In[38]:


loss


# In[39]:


plt.plot(loss)


# In[40]:


arr=model2.predict(x_test)


# In[72]:


arr


# In[42]:


arr1= model.predict(x_test)


# In[43]:


arr1


# In[73]:


from sklearn.metrics import mean_absolute_error


# In[74]:


model2.evaluate(x_test, y_test)


# In[75]:


mean=df['age'].mean()
std= df['age'].std()


# In[76]:


std


# In[77]:


def destanderdise(x):
    return x*std+mean


# In[78]:


vectorized_function = np.vectorize(destanderdise)


# In[79]:


arr1= vectorized_function(arr)


# In[80]:


arr1


# In[106]:


image_path="girl.jpg"
img= tf.keras.preprocessing.image.load_img(image_path)


# In[107]:


image= tf.keras.preprocessing.image.img_to_array(img)
plt.imshow(image.astype(int))
print(image.shape)


# In[108]:


def preprocess(arr):
    arr= arr/255.
    arr= tf.image.resize(arr, (128,128))
    return arr


# In[109]:


arr5=preprocess(image)


# In[110]:


arr5= tf.expand_dims(arr5, axis=0)


# In[111]:


arr5.shape


# In[112]:


value1= model2.predict(arr5)


# In[ ]:





# In[113]:


destanderdise(value1)


# In[ ]:





# In[ ]:




