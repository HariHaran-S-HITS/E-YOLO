import time

from imutils import paths
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
import os


# Function to Extract features from the images
def image_feature(direc):
    model = InceptionV3(weights='imagenet', include_top=False)
    features = [];
    img_name = [];
    for i in tqdm(direc):
        img = image.load_img(i, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feat = model.predict(x)
        feat = feat.flatten()
        features.append(feat)
        img_name.append(i)
    return features, img_name


def clustering():
    imagePaths = list(paths.list_images("./Data/Un-Labelled/Un-Clustered/"))
    features, image_name = image_feature(imagePaths)
    print("[INFO] clustering...")

    # Creating Clusters
    k = 2
    clusters = KMeans(k, random_state=40)
    clusters.fit(features)

    for label in clusters.labels_:
        parent_dir = f"Data/Un-Labelled/Cropped/"
        path = os.path.join(parent_dir, f"{now}/{label}")

        if os.path.exists(path):
            os.mkdir(path)


def cluster():
    print("[INFO] : Clustering Images ...")


if __name__ == "__main__":
    global now
    now = time.time()
    clustering()
