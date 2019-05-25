from sklearn.cluster import KMeans
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
import shutil

train_df = pd.read_csv("input/train_relationships.csv")

# compute the minimum number of images per person
min_images_per_person = 10000
min_folder = ''
for family_path in glob.glob('input_augmented/train/*/'):
    for person_path in glob.glob(family_path + '*/'):
        if len(glob.glob(person_path + '*.jpg')) < min_images_per_person:
            min_images_per_person = len(glob.glob(person_path + '*.jpg'))
            min_folder = person_path
min_images_per_person

# remove each empty folder
for family_path in glob.glob('input/train/*/'):
    for person_path in glob.glob(family_path + '*/'):
        if len(glob.glob(person_path + '*.jpg')) == 0:
            shutil.rmtree(person_path)

# augmentation
min_images_per_person = 10
for family_path in glob.glob('input_augmented/train/*/'):
    for person_path in glob.glob(family_path + '*/'):
        if len(glob.glob(person_path + '*.jpg')) < min_images_per_person:
            images = glob.glob(person_path + '*.jpg')
            n_images = len(images)

            for i in range(n_images, min_images_per_person + 1):
                file_name = images[np.random.choice(range(n_images))]
                img = Image.open(file_name)
                new_file_name = file_name.replace('.jpg', str(i) + '.jpg')
                if np.random.choice([True, False]):
                    new_img = mirror(img)
                else:
                    new_img = img
                new_img.save(new_file_name, 'JPEG')
            img.close()
