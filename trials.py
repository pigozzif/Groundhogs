# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

from sklearn.cluster import KMeans
from keras import Sequential, Model
from keras.layers import Conv2D, UpSampling2D, MaxPool2D, Dense, Input
import glob
import keras
import PIL
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


train_df = pd.read_csv("../input/train_relationships.csv")

#path = '../input/train/' + train_df.p1[100]
image_datas = list()
for path in train_df.p1:
    image_datas.extend([PIL.Image.open(f).resize((64, 64), PIL.Image.ANTIALIAS)
                        for f in glob.glob('../input/train/' + path + "/*.jpg", recursive=True)])

f, ax = plt.subplots(1, 4, figsize=(50, 20))
for i in range(4):
    ax[i].imshow(image_datas[i])

image_datas = image_datas[:1000]
for i in range(len(image_datas)):
    image_datas[i] = np.asarray(image_datas[i]) / 255.


encoder = Sequential([
    Conv2D(filters=64, kernel_size=8, strides=2, activation="relu", padding="same"),
    MaxPool2D(pool_size=4, padding="same"),
    Conv2D(filters=32, kernel_size=4, strides=1, activation="relu", padding="same"),
    MaxPool2D(pool_size=2, padding='same'),
    Dense(256),
    Dense(128),
])
decoder = Sequential([
    Dense(128),
    Dense(256),
    Conv2D(filters=32, kernel_size=4, strides=1, activation="relu", padding="same"),
    UpSampling2D(size=2),
    Conv2D(filters=64, kernel_size=8, strides=2, activation="relu", padding="same"),
    UpSampling2D(16),
    Conv2D(3, kernel_size=16, activation="sigmoid", padding="same")
])
autoencoder = Sequential([encoder, decoder])

autoencoder.compile(loss="mse", optimizer="adam",
                    metrics=["mean_squared_error"])

batch_size = 32
epochs = 5
learning_rate = 0.01

autoencoder.fit(image_datas.reshape(-1, 64, 64, 3), image_datas.reshape(-1, 64, 64, 3), batch_size, epochs, validation_split=0.2)

autoencoder.summary()

decoder.summary()


pred = autoencoder.predict(image_datas[:4], batch_size)
f, ax = plt.subplots(2, 4, figsize=(50, 20))
for i in range(4):
    ax[0][i].imshow(image_datas[i])
    ax[1][i].imshow(pred[i])
