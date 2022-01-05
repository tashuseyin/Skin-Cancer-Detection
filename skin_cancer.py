# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 11:20:42 2022

@author: machine
"""

# import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import os 
from glob import glob

warnings.filterwarnings("ignore")

from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

pd.set_option('display.max_columns', None)

#%%

# Load and Check data
data = pd.read_csv(os.path.join("dataset","HAM10000_metadata.csv"))
print(data.head())

print(data.shape)

#%%

# Veri setini düzenleme ve resimleri ekleme


imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join("dataset", '*', '*.jpg'))}
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

data["path"] = data["image_id"].map(imageid_path_dict.get)
data["cell_type"] = data["dx"].map(lesion_type_dict.get)
data["cell_type_idx"] = pd.Categorical(data["cell_type"]).codes

#%%


# Veri setinde missing value kontrolu
print(data.isnull().sum()) # 57 tane age sutununda eksik veri var.

# age sutunun dagılımına bakalım.
sns.displot(data["age"], kde=True, color="red")
plt.show()


# age sutununun istatistiklerine bakalım.
print(data["age"].describe())

# age sutunun medyan ve mean degeri birbirine yakın bu yuzden bu degerleri mean ile doldurabiliriz. 
data["age"].fillna(data["age"].mean(), inplace = True)
print(data.isnull().sum())

# veri setinde eksik veri kalmadı.

#%%

# Veri seti hakkında bilgi
data.info()

#%%

# Veri Görselleştirme

# Gender Distribution
x_gender = data["sex"].value_counts().index
y_gender = data["sex"].value_counts()
sns.barplot(x= x_gender, y= y_gender)
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Value")
plt.show()



# her bir hastalık türünün dağılımı
x_lesion_type  = data["cell_type"].value_counts().index
y_lesion_value = data["cell_type"].value_counts()
sns.barplot(x=x_lesion_type, y= y_lesion_value)
plt.title("Lesion Distribution")
plt.xlabel("Lesion Type")
plt.ylabel("Value")
plt.xticks(rotation=90)
plt.show()


# Dx Dağılımı
x_dx_type  = data["dx_type"].value_counts().index
y_dx_type_value = data["dx_type"].value_counts()
sns.barplot(x=x_dx_type, y= y_dx_type_value)
plt.title("Dx Distribution")
plt.xlabel("Dx Type")
plt.ylabel("Value")
plt.xticks(rotation=90)
plt.show()


# Localization Dağılımı
x_localization_type  = data["localization"].value_counts().index
y_localization_value = data["localization"].value_counts()
sns.barplot(x=x_localization_type, y= y_localization_value)
plt.title("Localization Distribution")
plt.xlabel("Localization Type")
plt.ylabel("Value")
plt.xticks(rotation=90)
plt.show()

#%%

# Resmi yükleme ve yeniden boyutlandırma
from PIL import Image
data["image"] = data["path"].map(lambda x: np.asarray(Image.open(x).resize((100,75))))

#%%

# görüntünün boyutu ve renk kanalı
# görüntü boyutu renk kanalı ve kac satır oldugu
print(data["image"].map(lambda x: x.shape).value_counts())

#%%

# train test split  

X = np.asarray(data["image"].tolist())
y = data["cell_type_idx"]


# Standardizasyon
X_mean = np.mean(X)
X_std = np.std(X)
X = (X - X_mean) / X_std

X_train , X_test, y_train, y_test = train_test_split(X, y, 
                                                     test_size= 0.20,
                                                     random_state=42)

from keras.utils.np_utils import to_categorical
# one hot encoding
y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

#%%

# CNN (Convolutional neural network)
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from keras import callbacks

input_shape = (75, 100, 3)
num_classes = 7

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',padding = 'Same',input_shape=input_shape))
model.add(Conv2D(64,kernel_size=(3, 3), activation='relu',padding = 'Same',))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), activation='relu',padding = 'Same'))
model.add(Conv2D(32, (3, 3), activation='relu',padding = 'Same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.40))

model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

optimizer = Adam(lr=0.001)
# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

#%%


from keras.preprocessing.image import ImageDataGenerator

# Data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # dimension reduction
    rotation_range=5,  # randomly rotate images in the range 5 degrees
    zoom_range=0.1,  # Randomly zoom image 10%
    width_shift_range=0.1,  # randomly shift images horizontally 10%
    height_shift_range=0.1,  # randomly shift images vertically 10%
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(X_train)

#%%

epochs = 50
batch_size = 10

earlystopping = callbacks.EarlyStopping(monitor='val_loss',
                                        mode='min',
                                        verbose=1,
                                        patience=20)

history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                              epochs=epochs, validation_data=(X_test, y_test),
                              steps_per_epoch=X_train.shape[0] // batch_size,
                              callbacks=[earlystopping])

#%%

# summarize history for acc
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#%%

# Modelin Test edilmesi

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

y_test_classes = np.argmax(y_test, axis=1)

# accuracy score
print("Test Score: {}".format(accuracy_score(y_test_classes, y_pred_classes)))


# confusion matrix
conf_mat = confusion_matrix(y_test_classes, y_pred_classes)
# plot the confusion matrix
f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(conf_mat, annot=True, linewidths=0.01, cmap="Greens", linecolor="gray", fmt='.1f', ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# classification report
print(classification_report(y_test_classes, y_pred_classes))

#%%

# save model
model.save("kerasmodel.h5")

#%%

# convert model tflite
import tensorflow as tf
from keras.models import load_model

model = load_model("keras_model.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
# Save the tflite model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
















