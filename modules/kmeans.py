import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
from math import sqrt
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

N_CLUSTER = 2

def normalise(img):
    return (img - np.mean(img))/(np.std(img))


def create_segmented_and_variance_images(im, w):
    (y, x) = im.shape

    image_variance = np.zeros(im.shape)
  
    for i in range(0, x, w):
        for j in range(0, y, w):
            box = [i, j, min(i + w, x), min(j + w, y)]
            block_stddev = np.std(im[box[1]:box[3], box[0]:box[2]])
            image_variance[box[1]:box[3], box[0]:box[2]] = block_stddev

      
    return image_variance


def k_means(image, image_variance):
    n_cluster = 2
    img_shape = image.shape

    #create train data(pixel value + image_variance), need to be in form of (n_samples, n_features)
    connected = np.stack((image.flatten(), image_variance.flatten()), axis=-1)
    kmeans = KMeans(n_clusters = N_CLUSTER)
    kmeans.fit(connected)
    prediction = kmeans.predict(connected)

    apply_predict = image.flatten()
    check = find_background_cluster(prediction, img_shape) 
    for i in range(len(prediction)):
        if prediction[i] == check:
            apply_predict[i] = 255

    final_image = apply_predict.reshape(image.shape)
    return final_image


#get number of background cluster(take clusters from corners)
def find_background_cluster(prediction, img_shape):
    ld_corner = ((img_shape[0] * img_shape[1]) - img_shape[1]) - 1
    check_corners = [prediction[0], prediction[-1], prediction[img_shape[1]], prediction[ld_corner]]
    count = 0

    for corner in check_corners:
        if corner == 0:
            count += 1
    if count == 2:
        return prediction[0]
    elif count > 2:
        return 0
    else:
        return 1


def create_image_paths(image_dir):
    image_paths = []
    for filename in os.listdir(image_dir):
        if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
            image_paths.append(os.path.join(image_dir, filename))

    return image_paths


def main(img_path):
    block_size = 16

    image = cv.imread(img_path, 0)
    image_variance = create_segmented_and_variance_images(image, block_size)
    return k_means(image, image_variance)
