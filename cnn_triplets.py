from keras import Sequential, Model
from keras.layers import Conv2D, UpSampling2D, MaxPool2D, Dense, Reshape, Flatten, merge, Input
from keras.losses import binary_crossentropy, mse
from keras.optimizers import Adam
from keras import backend as K
import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

# n_images =  22120
# n_people = 6948

train_df = pd.read_csv("input/train_relationships.csv")

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
    image_datas[i] = image_datas[i].resize((64, 64), Image.ANTIALIAS).convert('L')

for i in range(len(image_datas)):
    image_datas[i] = np.asarray(image_datas[i]) / 255.
image_datas = np.array(image_datas).reshape(-1, 64, 64, 1)


n_triplets_pp = 3


def preprocess_image(img):
    return np.asarray(img.resize((64, 64), Image.ANTIALIAS).convert('L')).reshape(64, 64, 1) / 255


def generate_triplets(maximum=1000):
    j = 0
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
                while True:
                    try:
                        file_name = np.random.choice(glob.glob('input_augmented/train/' + np.random.choice(list(train_df['p1'])) + '\\*.jpg'))
                        img = Image.open(file_name)
                        triplet.append(preprocess_image(img))

                        # triplet = np.asarray(triplet)

                        break
                    except:
                        pass

                yield triplet
                j += 1
                if j >= maximum:
                    return


imgs = [i for i in generate_triplets(20000)]
inpt = list(zip(*imgs))
inpt = [np.array(img).reshape(-1, 64, 64, 1) for img in inpt]
otpt = np.array(imgs).reshape(-1, 64, 64, 3)


def triplet_loss(y_true, y_pred):
    anchor = y_pred[:, :, :, :, 0]
    positive = y_pred[:, :, :, :, 1]
    negative = y_pred[:, :, :, :, 2]
    pos_dist = mse(anchor, positive)
    neg_dist = mse(anchor, negative)
    basic_loss = pos_dist - neg_dist + K.constant(0.1)
    loss = K.maximum(basic_loss, K.constant(0))
    return loss


# build the model
input_shape = (64, 64, 1)

inputs = [Input(input_shape) for _ in range(3)]

encoder = Sequential([
    Conv2D(filters=32, kernel_size=8, strides=1, activation="relu", padding="same", input_shape=input_shape),
    MaxPool2D(pool_size=2, padding="same"),
    Conv2D(filters=64, kernel_size=6, strides=1, activation="relu", padding="same"),
    MaxPool2D(pool_size=2, padding="same"),
    Conv2D(filters=128, kernel_size=2, strides=1, activation="relu", padding="same"),
    MaxPool2D(pool_size=2, padding="same"),
    Flatten(),
])
decoder = Sequential([
    Reshape((8, 8, 128)),
    UpSampling2D(2),
    Conv2D(filters=64, kernel_size=2, strides=1, activation="relu", padding="same"),
    UpSampling2D(2),
    Conv2D(filters=32, kernel_size=6, strides=1, activation="relu", padding="same"),
    UpSampling2D(2),
    Conv2D(filters=1, kernel_size=8, strides=1, activation="sigmoid", padding="same")
])
autoencoder = Sequential([encoder, decoder])

concatened = Reshape((64, 64, 1, 3))(merge.concatenate([autoencoder(input) for input in inputs], axis=-1))

triplet = Model(inputs=inputs, output=concatened)

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)

triplet.compile(loss=triplet_loss, optimizer=optimizer, metrics=[triplet_loss])

batch_size = 32
epochs = 100

history = triplet.fit(inpt, otpt, batch_size, epochs)

# show the loss plot
plt.figure(figsize=(10, 10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# show the predictions for 4 random people
n_img = 4
ind = np.random.choice(range(len(image_datas)), n_img)
pred = autoencoder.predict(image_datas[ind], batch_size)
f, ax = plt.subplots(2, n_img, figsize=(10, 10))
for i in range(n_img):
    ax[0][i].imshow(image_datas[ind[i]].reshape(64, 64), cmap='gray')
    ax[1][i].imshow(pred[i].reshape(64, 64), cmap='gray')
