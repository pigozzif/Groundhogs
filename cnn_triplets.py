import keras
from keras import Sequential, Model
from keras.layers import Conv2D, UpSampling2D, MaxPool2D, Dense, Reshape, Flatten
from keras import backend as K
import glob
from PIL import Image
from PIL.ImageOps import mirror
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import random


train_df = pd.read_csv("input/train_relationships.csv")

n_triplets_pp = 4


def preprocess_image(img):
    return np.asarray(img.resize((64, 64), Image.ANTIALIAS).convert('L')) / 255


def generate_triplets():
    for family_path in glob.glob('input_augmented/train/*/'):
        for person_path in glob.glob(family_path + '*/'):
            images = glob.glob(person_path + '*.jpg')
            random.shuffle(images)
            for i in range(n_triplets_pp):
                triplet = []
                # original image
                file_name = images.pop()
                img = Image.open(file_name)
                triplet.append(preprocess_image(img))

                # positive
                file_name = images.pop()
                img = Image.open(file_name)
                triplet.append(preprocess_image(img))

                # negative
                file_name = np.random.choice(glob.glob('input_augmented/train/' + np.random.choice(list(train_df['p1'])) + '\\*.jpg'))
                img = Image.open(file_name)
                triplet.append(preprocess_image(img))

                yield triplet


def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred
    pos_dist = K.reduce_sum(K.square(K.subtract(anchor, positive)), axis=-1)
    neg_dist = K.reduce_sum(K.square(K.subtract(anchor, negative)), axis=-1)
    basic_loss = K.add(K.subtract(pos_dist, neg_dist), alpha)
    loss = K.reduce_sum(K.maximum(basic_loss, 0))
    return loss


M = 1000
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
    Conv2D(filters=32, kernel_size=18, strides=1, activation="relu", padding="same", input_shape=(64, 64, 3)),
    MaxPool2D(pool_size=2, padding="same"),
    Conv2D(filters=64, kernel_size=11, strides=1, activation="relu", padding="same"),
    MaxPool2D(pool_size=2, padding="same"),
    Conv2D(filters=128, kernel_size=3, strides=1, activation="relu", padding="same"),
    MaxPool2D(pool_size=2, padding="same"),
    Flatten(),
    Dense(200),
    Dense(100),
    Dense(20),
    Dense(8)
])
decoder = Sequential([
    Dense(20, input_shape=(1, 8)),
    Dense(100),
    Dense(200),
    Dense(8192),
    Reshape((8, 8, 128)),
    UpSampling2D(2),
    Conv2D(filters=64, kernel_size=3, strides=1, activation="relu", padding="same"),
    UpSampling2D(2),
    Conv2D(filters=32, kernel_size=11, strides=1, activation="relu", padding="same"),
    UpSampling2D(2),
    Conv2D(filters=3, kernel_size=18, strides=1, activation="sigmoid", padding="same")
])
autoencoder = Sequential([encoder, decoder])

autoencoder.compile(loss="binary_crossentropy", optimizer="adam", metrics=["binary_crossentropy"])

batch_size = 32
epochs = 10
learning_rate = 0.1

history = autoencoder.fit(image_datas.reshape(image_datas.shape), image_datas.reshape(-1, 64, 64, 3), batch_size, epochs, validation_split=0.2, verbose=1)

plt.figure(figsize=(10,10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

encoder.summary()
decoder.summary()

n_img = 4
ind = np.random.choice(range(len(image_datas)), n_img)
pred = autoencoder.predict(image_datas[ind], batch_size)

f, ax = plt.subplots(2, n_img, figsize=(10, 10))
for i in range(n_img):
    ax[0][i].imshow(image_datas[ind[i]])
    ax[1][i].imshow(pred[i])

    # import winsound
    # frequency = 440  # Set Frequency To 2500 Hertz
    # duration = 1000  # Set Duration To 1000 ms == 1 second
    # winsound.Beep(frequency, duration)
    # winsound.Beep(440, 1000)
