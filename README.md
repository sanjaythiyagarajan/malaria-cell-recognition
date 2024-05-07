Exp.No : 04 

Date : 29.04.2024 
<br>

# Deep Neural Network for Malaria Infected Cell Recognition

## AIM

To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.

## Problem Statement and Dataset

- The problem at hand is the automatic classification of red blood cell images into two categories: parasitized and uninfected.
  - Malaria-infected red blood cells, known as parasitized cells, contain the Plasmodium parasite
  - uninfected cells are healthy and free from the parasite.
- The goal is to build a convolutional neural network (CNN) model capable of accurately distinguishing between these two classes based on cell images.
- Traditional methods of malaria diagnosis involve manual inspection of blood smears by trained professionals, which can be time-consuming and error-prone.
- Automating this process using deep learning can significantly speed up diagnosis, reduce the workload on healthcare professionals, and improve the accuracy of detection.
- Our dataset comprises 27,558 cell images, evenly split between parasitized and uninfected cells.
- These images have been meticulously collected and annotated by medical experts, making them a reliable source for training and testing our deep neural network.

## Neural Network Model

<p align="center">

</p>

## DESIGN STEPS

- **Step 1:** We begin by importing the necessary Python libraries, including TensorFlow for deep learning, data preprocessing tools, and visualization libraries.
- **Step 2:** To leverage the power of GPU acceleration, we configure TensorFlow to allow GPU processing, which can significantly speed up model training.
- **Step 3:** We create an image generator that performs data augmentation, including rotation, shifting, rescaling, and flipping. Data augmentation enhances the model's ability to generalize and recognize malaria-infected cells in various orientations and conditions.
- **Step 4:** We design a convolutional neural network (CNN) architecture consisting of convolutional layers, max-pooling layers, and fully connected layers. The model is compiled with appropriate loss and optimization functions.
- **Step 5:** We split the dataset into training and testing sets, and then train the CNN model using the training data. The model learns to differentiate between parasitized and uninfected cells during this phase.
- **Step 6:** We evaluate the trained model's performance using the testing data, generating a classification report and confusion matrix to assess accuracy and potential misclassifications.

## PROGRAM

> Developed by: SANJAY T <br>
> Register no: 212222110039

**Importing libraries**
```python
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True 
config.log_device_placement = True 
sess = tf.compat.v1.Session(config=config)
set_session(sess)
```
**To share the GPU resources for multiple sessions**
```python
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from sklearn.metrics import classification_report,confusion_matrix
```

**Loading dataset**
```python
my_data_dir = './dataset/cell_images'
os.listdir(my_data_dir)
```
```python
test_path = my_data_dir+'/test/'
train_path = my_data_dir+'/train/'```
```python
os.listdir(train_path)
```
```python
len(os.listdir(train_path+'/uninfected/'))
len(os.listdir(train_path+'/parasitized/'))
```


```py
para_img= imread(train_path+
                 '/parasitized/'+
                 os.listdir(train_path+'/parasitized')[0])
```

```py
print('Developed By : SANJAY T [212222110039]')
plt.imshow(para_img)
```

```py
dim1 = []
dim2 = []
for image_filename in os.listdir(test_path+'/uninfected'):
    img = imread(test_path+'/uninfected'+'/'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)
```

```py
sns.jointplot(x=dim1,y=dim2)
```

**Generating images**
```python
image_shape = (130,130,3)
help(ImageDataGenerator)
image_gen = ImageDataGenerator(rotation_range=20,
                               width_shift_range=0.10, 
                               height_shift_range=0.10, 
                               rescale=1/255, 
                               shear_range=0.1, 
                               zoom_range=0.1, 
                               horizontal_flip=True,
                               fill_mode='nearest')
```

```py
image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)
```

**Network model**
```python
model = models.Sequential()
model.add(layers.Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=32, kernel_size=(5,5), activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))


model.add(layers.Flatten())

model.add(layers.Dense(128))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))


model.add(layers.Dense(64))
model.add(layers.Activation('relu'))

model.add(layers.Dense(1))
model.add(layers.Activation('sigmoid'))

```
```py
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
```
```py
print('Developed : SANJAY T [212222110039]')
model.summary()
```


```python
batch_size = 16
help(image_gen.flow_from_directory)
train_image_gen = image_gen.flow_from_directory(train_path,
                                               target_size=image_shape[:2],
                                                color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary')
```
```py
train_image_gen.batch_size
len(train_image_gen.classes)
train_image_gen.total_batches_seen

```
```py
test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle=False)
```
```python
train_image_gen.class_indices
results = model.fit(train_image_gen,epochs=4,
                              validation_data=test_image_gen
                             )
```

**Metrics**

```python
model.save('cell_model.h5')

losses = pd.DataFrame(model.history.history)

print("SANJAY T \n 212222110039")
losses[['loss','val_loss']].plot()
```
```python
model.metrics_names

model.evaluate(test_image_gen)

pred_probabilities = model.predict(test_image_gen)

test_image_gen.classes

predictions = pred_probabilities > 0.5

print("SANJAY T  \n 212222110039")
print(classification_report(test_image_gen.classes,predictions))
```
```python
print("SANJAY T \n 212222110039")
confusion_matrix(test_image_gen.classes,predictions)
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/sanjaythiyagarajan/malaria-cell-recognition/assets/119409242/4629ade3-4984-4203-90cb-0df3a112763f)


### Classification Report and Confusion Matrix


![image](https://github.com/sanjaythiyagarajan/malaria-cell-recognition/assets/119409242/7c800b09-1cb5-4666-a48f-baa305edde81)


### New Sample Data Prediction
```py
import random
import tensorflow as tf

list_dir=["UnInfected","parasitized"]
dir_=(list_dir[1])
para_img= imread(train_path+
                 '/'+dir_+'/'+
                 os.listdir(train_path+'/'+dir_)[random.randint(0,100)])
img  = tf.convert_to_tensor(np.asarray(para_img))
img = tf.image.resize(img,(130,130))
img=img.numpy()
pred=bool(model.predict(img.reshape(1,130,130,3))<0.5 )
plt.title("Model prediction: "+("Parasitized" if pred
    else "Un Infected")+"\nActual Value: "+str(dir_))
plt.axis("off")
print("SANJAY T 212222110039")
plt.imshow(img)
plt.show()
```

![image](https://github.com/sanjaythiyagarajan/malaria-cell-recognition/assets/119409242/d3c95f88-88ae-4d0a-931c-066ad4a3c254)



## RESULT

### Thus a deep neural network for Malaria infected cell recognition and to analyze the performance is created using tensorflow.
