import argparse

import sys
import os

import cv2
import imghdr

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split


# Global constants
EPOCHS = 15
IMG_HEIGHT = 40
IMG_WIDTH = 40
NUM_CATEGORIES = 10        # Allows user to limit number of categories
IMAGES_PER_CATEGORY = 1000  # Allows user to limit max number of images per category
# to train models more quickly.
TEST_SIZE = 0.2            # Allows user to modify proportion of data set
# withheld for testing.


def main():

    args = parse_input()

    # Load images and labels for all image files
    images, labels = load_data(args.training_data_directory)

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate the performance of the neural network
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if args.save:
        filename = args.save
        model.save(filename)
        print(f"Model saved to {filename}.")


def parse_input():
    parser = argparse.ArgumentParser(
        description="Train and/or test the CNN on the GTSRB dataset"
    )
    parser.add_argument(
        '-l', '--load',
        help='Specify a file [*.h5] to load as a pretrained model'
    )
    parser.add_argument(
        '-s', '--save',
        help='Specify a file [*.h5] to save the trained model')
    parser.add_argument(
        '-t', '--train',
        action='store_true',
        help='boolean value that specifies whether or not to train the model.  If no model is loaded, this defaults to true; if a model is loaded, this argument defaults to false.'
    )
    parser.add_argument(
        'training_data_directory',
        type=str,
        help='the directory where the training data can be found.  Images must be in .ppm format and should be in labelled subdirectories "[training_data_directory/Images/[label]/"'
    )
#    parser.add_argument('testing_data_directory', type = str, help = 'the directory where the testing data can be found.  Images must be in .ppm format and should be in a subdirectory "[testing_data_directory/Images/"')

    args = parser.parse_args()

    if not args.load:
        args.train = True

    # For testing purposes
    print(args.load)
    print(args.save)
    print(args.train)
    print(args.training_data_directory)
    # print(args.testing_data_directory)
    return args


def load_data(data_dir):
    """
    Load image data from directory 'data_dir'.

    `data_dir` must have a sub-directory called 'Images', which in turn must have
    directories named after each category, numbered 0 through NUM_CATEGORIES - 1.
    Inside each category directory will be some number of .ppm image files (names of
    image files do not matter).  Files with different extensions will be ignored.

    The function returns a tuple `(images, labels)` which is a list of all the
    images in the data directory, where each image is formatted as a numpy ndarray
    with dimensions IMG_WIDTH x IMG_HEIGHT x 3.  `labels` is a list of integer labels,
    representing the categories for each of the corresponding `images`.
    """

    images, labels = [], []
    im_dir = os.path.join(data_dir, 'Images')
    for i, directory in enumerate(os.listdir(im_dir)):
        if i >= NUM_CATEGORIES:
            break
        cat_dir = os.path.join(im_dir, directory)
        
        j = 0
        for fil in os.scandir(cat_dir):
            if j >= IMAGES_PER_CATEGORY:
                break
            if imghdr.what(fil) == 'ppm':
                image = cv2.imread(fil.path)
                images.append(cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT)))
                labels.append(i)
                j+= 1

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. The 'input_shape'
    of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.

    The output layer will have `NUM_CATEGORIES` neurons; one for each category.
    """
    return get_project_model()


def get_project_model():
    """
    Returns model used for the submission of my cs-50 project.

    Performance statistics when trained for 15 epochs,
    accuracy ~ 95-97 %
    """

    model = tf.keras.models.Sequential()

    # add convolutional layer
    model.add(tf.keras.layers.Conv2D(
        32,  # number of cells
        (3, 3),  # kernel size
        activation="relu",
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    ))

    # add maxpooling layer
    model.add(tf.keras.layers.MaxPooling2D(
        pool_size=(3, 3)
    ))

    # Flatten
    model.add(tf.keras.layers.Flatten())

    # add hidden layer with dropout
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES * 32, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.2))

    # Add hidden layer
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES * 16, activation="relu"))

    # Add output layer
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
