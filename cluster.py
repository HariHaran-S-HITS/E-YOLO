import numpy as np
from imutils import paths
from sklearn.cluster import DBSCAN


def get_images():
    imagePaths = list(paths.list_images("./Data/Un-Labelled/Un-Clustered/"))
    data = []
    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # load the input image and convert it from RGB (OpenCV ordering)
        # to dlib ordering (RGB)
        print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
        print(imagePath)
        with open(imagePath, "rb") as image:
            # encodings = face_recognition.face_encodings(rgb, boxes)
            d = [{"imagePath": imagePath, "image": image.read}]
            data.extend(d)

    return data


def clustering():
    data = get_images()
    data = np.array(data)
    encodings = [d["encoding"] for d in data]

    print("[INFO] clustering...")
    clt = DBSCAN(metric="euclidean", n_jobs=-1, min_samples=5)

    try:
        breakpoint()
        clt.fit(encodings)

        # determine the total number of unique faces found in the dataset
        # labelIDs = np.unique(clt.labels_)
        # numUniqueObjects = len(np.where(labelIDs > -1)[0])

        # print("[INFO] # unique objects: {}".format(numUniqueObjects))
    except Exception as e:
        print(e)
        print("[INFO] Clustering Cancelled due lack of images")


def cluster():
    print("[INFO] : Clustering Images ...")


if __name__ == "__main__":
    cluster()
