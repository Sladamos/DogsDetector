import os

import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array, image_utils
from matplotlib import pyplot as plt, image as mpimg
from numpy import expand_dims, resize

from displayer.ImagesDisplayer import ImagesDisplayer


class DogsImagesDisplayer(ImagesDisplayer):

    def __init__(self):
        self.datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.05,
            rotation_range=10,
            zoom_range=0.25,
            horizontal_flip=True,
            validation_split=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2)

        self.img_width, self.img_height = 224, 224
        self.channels = 3
        self.num_images = 72
        self.image_arr_size = self.img_width * self.img_height * self.channels

    def display_images(self):
        images = self.get_images(os.path.normpath('./images/dogs/All_images'))
        self.plot_images(images)

    def display_transformed_image(self):

        img = load_img(os.path.normpath('./images/dogs/Images/n02107683-Bernese_mountain_dog/n02107683_5018.jpg'))
        data = img_to_array(img)
        samples = expand_dims(data, 0)
        it = self.datagen.flow(samples, batch_size=1)

        for i in range(9):
            plt.subplot(330 + 1 + i)
            batch = it.next()
            image = batch[0]
            plt.imshow(image)

        plt.savefig(os.path.normpath('./newHope/augmentedDog.png'), transparent=False, bbox_inches='tight', dpi=900)
        plt.show()

    def display_image_with_gaussian_noise(self, path, std_dev):
        image = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (592, 333))
        noise = np.random.normal(loc=0, scale=std_dev, size=image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        images = np.concatenate((image, noisy_image), axis=0)
        cv2.imshow('Images', images)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def plot_images(self, instances, images_per_row=12):
        images_per_row = min(len(instances), images_per_row)
        n_rows = (len(instances) - 1) // images_per_row + 1
        row_images = []
        for row in range(n_rows):
            if row == len(instances) / images_per_row:
                break
            rimages = instances[row * images_per_row:(row+1) * images_per_row]
            row_images.append(np.concatenate(rimages, axis=1))
        image = np.concatenate(row_images, axis=0)
        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        plt.axis("off")
        plt.savefig(os.path.normpath('./newHope/dogsImages.png'), transparent=True, bbox_inches='tight', dpi=900)
        plt.show()

    def get_images(self, image_dir):
        images = []

        for type in os.listdir(image_dir)[:self.num_images]:
            type_images = os.listdir(image_dir + '/' + type)

            for image in type_images[:1]:
                image_file = os.path.join(image_dir, type + '/', image)
                img = load_img(image_file, target_size=(self.img_width, self.img_height), color_mode="rgb")
                images.append(img)
                print(type, ':', image)
        return images
