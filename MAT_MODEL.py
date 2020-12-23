import json
import math
import os
import cv2
from PIL import Image
import numpy as np
from keras import layers
from keras.applications import ResNet50, MobileNet, DenseNet201, InceptionV3, NASNetLarge, InceptionResNetV2, NASNetMobile
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
from tqdm import tqdm
from keras import backend as K
import gc
from functools import partial
from sklearn import metrics
from collections import Counter
import json
import itertools

# Датасет з малюнками
def Dataset_loader(DIR, RESIZE, sigmaX=10):
    IMG = []
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    for IMAGE_NAME in tqdm(os.listdir(DIR)):
        PATH = os.path.join(DIR, IMAGE_NAME)
        _, ftype = os.path.splitext(PATH)
        if ftype == ".png":
            img = read(PATH)

            img = cv2.resize(img, (RESIZE, RESIZE))

            IMG.append(np.array(img))
    return IMG

benign_train = np.array(Dataset_loader('data/train/benign', 224))
malign_train = np.array(Dataset_loader('data/train/malignant', 224))
benign_test = np.array(Dataset_loader('data/validation/benign', 224))
malign_test = np.array(Dataset_loader('data/validation/malignant', 224))

# Створення класів
benign_train_label = np.zeros(len(benign_train))
malign_train_label = np.ones(len(malign_train))
benign_test_label = np.zeros(len(benign_test))
malign_test_label = np.ones(len(malign_test))

# Об'єднанання даних
X_train = np.concatenate((benign_train, malign_train), axis = 0)
Y_train = np.concatenate((benign_train_label, malign_train_label), axis = 0)
X_test = np.concatenate((benign_test, malign_test), axis = 0)
Y_test = np.concatenate((benign_test_label, malign_test_label), axis = 0)

# Дані для навчання
s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
Y_train = Y_train[s]

# Дані для розпізнавання
s = np.arange(X_test.shape[0])
np.random.shuffle(s)
X_test = X_test[s]
Y_test = Y_test[s]

# Категорії
Y_train = to_categorical(Y_train, num_classes= 2)
Y_test = to_categorical(Y_test, num_classes= 2)

x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=11)
# Вивід визначених даних для навчання та класифікування
w=60
h=40
fig=plt.figure(figsize=(15, 15))
columns = 4
rows = 3

for i in range(1, columns*rows +1):
    ax = fig.add_subplot(rows, columns, i)
    if np.argmax(Y_train[i]) == 0:
        ax.title.set_text('Benign')
    else:
        ax.title.set_text('Malignant')
    plt.imshow(x_train[i], interpolation='nearest')
plt.show()

BATCH_SIZE = 16

train_generator = ImageDataGenerator(zoom_range=2, rotation_range = 90, horizontal_flip=True, vertical_flip=True)

#створення моделі
def build_model(backbone, lr=1e-4):
    model = Sequential()
    model.add(backbone)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])

    return model

#вивід створеної моделі
K.clear_session()
gc.collect()

resnet = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = build_model(resnet, lr=1e-4)
model.summary()

# Збереження моделі
learn_control = ReduceLROnPlateau(monitor='val_acc', patience=5, verbose=1,factor=0.2, min_lr=1e-7)
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

#навчання та оцінка
history = model.fit_generator(
    train_generator.flow(x_train, y_train, batch_size=BATCH_SIZE), steps_per_epoch=x_train.shape[0] / BATCH_SIZE, epochs=20, validation_data=(x_val, y_val), callbacks=[learn_control, checkpoint])

with open('history.json', 'w') as f:
    json.dump(str(history.history), f)

#залежність втрат від епохи
history_df = pd.DataFrame(history.history)
history_df[['acc', 'val_acc']].plot()

#залежність точності від епохи
history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()

#Прогнозування
model.load_weights("weights.best.hdf5")
Y_val_pred = model.predict(x_val)
accuracy_score(np.argmax(y_val, axis=1), np.argmax(Y_val_pred, axis=1))
Y_pred = model.predict(X_test)
tta_steps = 10
predictions = []

for i in tqdm(range(tta_steps)):
    preds = model.predict_generator(train_generator.flow(X_test, batch_size=BATCH_SIZE, shuffle=False), steps=len(X_test) / BATCH_SIZE)
    predictions.append(preds)
    gc.collect()

Y_pred_tta = np.mean(predictions, axis=0)

i = 0
prop_class = []
mis_class = []

for i in range(len(Y_test)):
    if (np.argmax(Y_test[i]) == np.argmax(Y_pred_tta[i])):
        prop_class.append(i)
    if (len(prop_class) == 8):
        break

i = 0
for i in range(len(Y_test)):
    if (not np.argmax(Y_test[i]) == np.argmax(Y_pred_tta[i])):
        mis_class.append(i)
    if (len(mis_class) == 8):
        break

# вивід 8 класифікованих зображень
w = 60
h = 40
fig = plt.figure(figsize=(18, 10))
columns = 4
rows = 2

def Transfername(namecode):
    if namecode == 0:
        return "Benign"
    else:
        return "Malignant"

for i in range(len(prop_class)):
    ax = fig.add_subplot(rows, columns, i + 1)
    ax.set_title("Predicted result:" + Transfername(np.argmax(Y_pred_tta[prop_class[i]])) + "\n" + "Actual result: " + Transfername(np.argmax(Y_test[prop_class[i]])))
    plt.imshow(X_test[prop_class[i]], interpolation='nearest')
plt.show()