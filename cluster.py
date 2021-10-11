import glob
import time

import cv2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
import os
import shutil


# Function to Extract features from the images
def image_feature(direc):
    model = InceptionV3(weights='imagenet', include_top=False)
    features = []
    img_name = []
    imgs = []
    for i in tqdm(direc):
        img = image.load_img(i, target_size=(224, 224))
        x = img_to_array(img)
        x_ = np.expand_dims(x, axis=0)
        x = preprocess_input(x_)
        feat = model.predict(x)
        feat = feat.flatten()
        features.append(feat)
        img_name.append(i)
        imgs.append(x_)
    return features, img_name, imgs


def clustering(now):
    imagePaths = glob.glob("./Data/Un-Labelled/Cropped/Un-Clustered/*")
    features, image_name, img = image_feature(imagePaths)
    print("[INFO] clustering...")

    # Creating Clusters
    k = 2
    clusters = KMeans(k, random_state=40)
    clusters.fit(features)

    for i in range(len(clusters.labels_)):
        parent_dir = f"Data/Un-Labelled/Cropped/Clustered"
        path = os.path.join(parent_dir, f"{now}")
        subpath = os.path.join(path, f"{clusters.labels_[i]}")

        if not os.path.exists(path):
            os.mkdir(path)
            if not os.path.exists(subpath):
                os.mkdir(subpath)

        shutil.copy(image_name[i], f"{subpath}")


def cluster():
    print("[INFO] : Clustering Images ...")


if __name__ == "__main__":
    now = time.time()
    clustering(now)
