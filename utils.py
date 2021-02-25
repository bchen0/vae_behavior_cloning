import os

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Cropping2D


def crop_single_image(image, grayscale):
    if grayscale:
        input_image_shape = (210, 160, 1)
        image = image[..., np.newaxis]
    else:
        input_image_shape = (210, 160, 3)

    cropper = Sequential()
    cropper.add(
        Cropping2D(cropping=((0, 38), (0, 0)), input_shape=input_image_shape))
    # cropping = ((top,bottom),(left,right))
    cropped_image = cropper.predict(image[np.newaxis, :])[0]

    if grayscale:
        cropped_image = cropped_image[..., 0]
    return cropped_image


def crop_single_image_rgb(image):
    return crop_single_image(image, False)


def crop_single_image_gray(image):
    return crop_single_image(image, True)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def get_all_files(directory):
    files = [f for f in os.listdir(directory) if
             os.path.isfile(os.path.join(directory, f))]
    return files

def display_img(img, grayscale=True, figsize=(20,15)):
    assert len(img.shape)
    img = (img * 255).astype(np.uint8)
    if not grayscale:
        raise NotImplementedError
    if grayscale:
        num_frames = img.shape[-1]
        fig, axs = plt.subplots(1, num_frames, figsize=figsize)
        for i in range(num_frames):
            axs[i].imshow(img[..., i], cmap='gray')
    return fig, axs
