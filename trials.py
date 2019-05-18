from sklearn.cluster import KMeans
from keras import Sequential, Model
from keras.layers import Conv2D, UpSampling2D, MaxPool2D, Dense, Reshape, Flatten
import glob
import keras
from keras import backend as K
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

train_df = pd.read_csv("input/train_relationships.csv")

M = 5000000
i = 0
image_datas = list()
for path in train_df.p1:
    for f in glob.glob('input/train/' + path + "/*.jpg", recursive=True):
        temp = Image.open(f)
        image = temp.copy()
        image_datas.append(image)
        temp.close()
        i += 1
        if i == M:
            break
    if i == M:
        break

np.random.shuffle(image_datas)

for i in range(len(image_datas)):
    image_datas[i] = image_datas[i].resize((64, 64), Image.ANTIALIAS)

for i in range(len(image_datas)):
    image_datas[i] = np.asarray(image_datas[i]) / 255.
image_datas = np.array(image_datas)

encoder = Sequential([
    Conv2D(filters=64, kernel_size=11, strides=2, activation="relu", padding="same"),
    MaxPool2D(pool_size=4, padding="same"),
    Conv2D(filters=128, kernel_size=3, strides=1, activation="relu", padding="same"),
    MaxPool2D(pool_size=2, padding='same'),
    Flatten(),
    Dense(200),
    Dense(100),
    Dense(20),
    Dense(8)
])
decoder = Sequential([
    Dense(20),
    Dense(100),
    Dense(200),
    Dense(2048),
    Reshape((4, 4, 128)),
    UpSampling2D(4),
    Conv2D(filters=64, kernel_size=3, strides=1, activation="relu", padding="same"),
    UpSampling2D(8),
    Conv2D(filters=3, kernel_size=11, strides=2, activation="sigmoid", padding="same"),
])
autoencoder = Sequential([encoder, decoder])

autoencoder.compile(loss="mse", optimizer="adam", metrics=["mean_squared_error"])

batch_size = 32
epochs = 200
learning_rate = 0.01

autoencoder.fit(image_datas.reshape(-1, 64, 64, 3), image_datas.reshape(-1, 64, 64, 3), batch_size, epochs, validation_split=0.2, verbose=2)

encoder.summary()
decoder.summary()

ind = np.random.choice(range(len(image_datas)), 6)
pred = autoencoder.predict(image_datas[ind], batch_size)

f, ax = plt.subplots(2, 6, figsize=(10, 10))
for i in range(6):
    ax[0][i].imshow(image_datas[ind[i]])
    ax[1][i].imshow(pred[i])
