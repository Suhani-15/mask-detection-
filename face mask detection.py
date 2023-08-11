#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


img_array=cv2.imread("I:/Face mask detection/data/with_mask/with_mask_1.jpg")


# In[4]:


plt.imshow(img_array)


# In[5]:


plt.imshow(cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB))


# In[6]:


img_array.shape


# In[7]:


Datadiretory="I:/Face mask detection/data"
Classes=["with_mask","without_mask"]
for category in Classes:
    path=os.path.join(Datadiretory,category)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img))
        plt.imshow(cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB))
        plt.show()
        break
    break


# In[8]:


img_size=224
new_array=cv2.resize(img_array,(img_size,img_size))
plt.imshow(cv2.cvtColor(new_array,cv2.COLOR_BGR2RGB))
plt.show()


# In[9]:


training_Data=[]
def create_training_Data():
    for category in Classes:
        path=os.path.join(Datadiretory,category)
        class_num=Classes.index(category)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img))
                new_array=cv2.resize(img_array,(img_size,img_size))
                training_Data.append([new_array,class_num])
            except Exception as e:
                pass


# In[10]:


create_training_Data()


# In[11]:


print(len(training_Data))


# In[12]:


import random
random.shuffle(training_Data)


# In[13]:


x=[]
y=[]
for features,label in training_Data:
    x.append(features)
    y.append(label)
x=np.array(x).reshape(-1,img_size,img_size,3)


# In[14]:


x.shape


# In[15]:


x=x/255.0;


# In[16]:


y[1000]


# In[17]:


y=np.array(y)


# In[18]:


import pickle
pickle_out =open("x.pickle","wb")
pickle.dump(x,pickle_out)
pickle_out.close()
pickle_out =open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()


# In[19]:


pickle_in =open("x.pickle","rb")
x=pickle.load(pickle_in)
pickle_in =open("y.pickle","rb")
y=pickle.load(pickle_in)


# In[20]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[21]:


model=tf.keras.applications.mobilenet.MobileNet()


# In[22]:


model.summary()


# In[23]:


base_input=model.layers[0].input


# In[24]:


base_output=model.layers[-4].output


# In[25]:


flat_layer=layers.Flatten()(base_output)
final_output=layers.Dense(1)(flat_layer)
final_output=layers.Activation('sigmoid')(final_output)


# In[26]:


new_model=keras.Model(inputs=base_input, outputs=final_output)


# In[27]:


new_model.summary()


# In[28]:


new_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


# In[29]:


new_model.fit(x,y,epochs=1,validation_split=0.1)


# In[30]:


new_model.save('my_model3.h5') 


# In[ ]:


new_model=tf.keras.models.load_model('my_model3.h5')


# ### checking network for predictions

# In[31]:


frame=cv2.imread("I:/Face mask detection/data/with_mask/with_mask_3.jpg")


# In[32]:


plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))


# In[33]:


final_image=cv2.resize(frame,(224,224))
final_image=np.expand_dims(final_image,axis=0)
final_image=final_image/255.0


# In[34]:


Predictions=new_model.predict(final_image)


# In[35]:


Predictions


# ### check network for unknown images

# In[36]:


frame=cv2.imread("I:/Face mask detection/sad image.jpeg")


# In[37]:


plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))


# In[38]:


faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')


# In[39]:


gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


# In[40]:


gray.shape


# In[48]:


faces=faceCascade.detectMultiScale(gray,1.1,4)
for x,y,w,h in faces:
    roi_gray=gray[y:y+h, x:x+w]
    roi_color=frame[y:y+h, x:x+w]
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    facess=faceCascade.detectMultiScale(roi_gray)
    if len(facess)==0:
        print("Face not detected")
    else:
        for (ex,ey,ew,eh) in facess:
            face_roi= roi_color[ey:ey+eh, ex:ex+ew]


# ## RealTime Video Demo

# In[ ]:


import cv2
path="haarcascade_frontalface_default.xml"
font_scale=1.5
font=cv2.FONT_HERSHEY_PLAIN
rectangle_bgr=(255,255,255)
img=np.zeros((500,500))
text="Some text in a box!"
(text_width,text_height)=cv2.getTextSize(text,font,fontScale=font_scale,thickness=1)[0]
text_offset_x=10
text_offset_y=img.shape[0]-25
#make the coords of the box with small padding of two pixels
box_coords=((text_offset_x,text_offset_y),(text_offset_x+text_width+2,text_offset_y-text_height-2))
cv2.rectangle(img,box_coords[0],box_coords[1],rectangle_bgr,cv2.FILLED)
cv2.putText(img,text,(text_offset_x,text_offset_y),font,fontScale=font_scale,color=(0,0,0),thickness=1)
cap=cv2.VideoCapture(1)
#check if webcam is opened
if not cap.isOpened():
    cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")
while True:
    ret,frame=cap.read()
    faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray,1.1,4)
    for x,y,w,h in faces:
        roi_gray=gray[y:y+h, x:x+w]
        roi_color=frame[y:y+h, x:x+w]
        cv2.rectangle(frame,(x,y), (x+w,y+h), (255,0,0),2)
        facess=faceCascade.detectMultiScale(roi_gray)
        if len(facess)==0:
            print("Face not detected")
        else:
            for (ex,ey,ew,eh) in facess:
                face_roi= roi_color[ey:ey+eh, ex:ex+ew]
    final_image=cv2.resize(face_roi,(224,224))
    final_image=np.expand_dims(final_image,axis=0)
    final_image=final_image/255.0
    font=cv2.FONT_HERSHEY_SIMPLEX
    Predictions=new_model.predict(final_image)
    font_scale=1.5
    font=cv2.FONT_HERSHEY_PLAIN
    if(Predictions):
        status="No Mask"
        x1,y1,w1,h1=0,0,175,75
        cv2.rectangle(frame,(x1,x1),(x1+w1, y1+h1),(0,0,0),-1)
        cv2.putText(frame,status,(x1+int(w1/10),y1+int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255),2)
        cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
        cv2.rectangle(frame,(x,y),(x+w, y+h),(0,0,255))
    else:
        status="Face Mask"
        x1,y1,w1,h1=0,0,175,75
        cv2.rectangle(frame,(x1,x1),(x1+w1, y1+h1),(0,0,0),-1)
        #add text
        cv2.putText(frame,status,(x1+int(w1/10),y1+int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255),2)
        cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
        cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,0))
    cv2.imshow('Face Mask Detection',frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
        


# In[ ]:




