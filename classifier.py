import argparse

import sys
import os

import csv
import cv2
import imghdr

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split


# Global constants
EPOCHS = 10
IMG_HEIGHT = 40
IMG_WIDTH = 40
IMAGES_PER_CATEGORY = 10000  # Allows user to limit max number of images per category
# to train models more quickly.
TEST_SIZE = 0.2              # Allows user to modify proportion of data set
# withheld for testing.


def main():

    args = parse_input()

    # get a compiled neural network
    model = get_model(args.load, args.num_categories)

    # If no number of categories was specified as an input,
    # set this value to the number of outputs of the loaded (or new)
    # model
    if not args.num_categories:
        args.num_categories = model.output.shape[1]
        
    if not args.testing_data_directory:
        
        # Load images and labels for entire training set
        images, labels = load_training_data(
            args.training_data_directory,
            args.num_categories
        )

        # Split data into training and testing sets
        labels = tf.keras.utils.to_categorical(labels)
        x_train, x_test, y_train, y_test = train_test_split(
            np.array(images), np.array(labels), test_size=TEST_SIZE
        )

    elif args.testing_data_directory:
        # Load images and labels for entire training set
        training_images, training_labels = load_training_data(
            args.training_data_directory,
            args.num_categories
        )
        training_labels = tf.keras.utils.to_categorical(training_labels)
        x_train, y_train = np.array(training_images), np.array(training_labels)

        # Load images and labels for entire testing set
        testing_images, testing_labels = load_testing_data(
            args.testing_data_directory,
            args.num_categories
        )
        testing_labels = tf.keras.utils.to_categorical(testing_labels)
        x_test, y_test = np.array(testing_images), np.array(testing_labels)
            

    # Fit model on training data
    if args.train:
        model.fit(x_train, y_train, epochs=EPOCHS)
    else:
        print("Model is not being trained")

    # Evaluate the performance of the neural network
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if args.save:
        filename = args.save
        model.save(filename)
        print(f"Model saved to {filename}.")
    return None

def parse_input():
    parser = argparse.ArgumentParser(
        description="Train and/or test the CNN on the GTSRB dataset.  When no options are specified, the program will load data from the training_data_directory; then split into training/testing sets; train the model; then evaluate the trained model on the testing set.  If --testing_data_directory is specified, the program will train on the entire training set and evaluate against the testing dataset.  This is useful for comparisons of models; as the training and testing sets are always the same.  The save/load options allow the user to load a pretrained model or save the trained model; if --load is provided, then the model will not be trained (only evaluated) unless the --train argument is given.  The default behavior accepts a training_data_directory, "
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
        help='boolean value that specifies whether or not to train the model.  If no model is loaded, this defaults to true; if a model is loaded, this argument defaults to false unless supplied as a command line argument.'
    )
    parser.add_argument(
        '-c', '--num_categories',
        type = int,
        help='Specify the number of categories of images.  This defaults to 43 (the number of categories in the GTSRB dataset), unless a model is loaded, in which case this defaults to the correct number of categories for the loaded model.  If this option is specified and conflicts with a model that is loaded, the program will exit with an error.'
    )
    parser.add_argument(
        'training_data_directory',
        type=str,
        help='the directory where the training data can be found.  Images must be in .ppm format and should be in labelled subdirectories "[training_data_directory/Images/[label]/"'
    )
    parser.add_argument(
        '--testing_data_directory', '-td',
        type = str,
        help = 'the directory where the testing data can be found.  Images must be in .ppm format and should be in a subdirectory "[testing_data_directory/Images/".  Label data should be in a csv file in the same directory.'
    )

    args = parser.parse_args()

    if not args.load:
        args.train = True

    # For testing purposes
    # print(args.load)
    # print(args.save)
    # print(args.train)
    # print(args.training_data_directory)
    # print(args.testing_data_directory)
    return args


def load_training_data(
        data_dir,
        num_categories,
        images_per_category = IMAGES_PER_CATEGORY):
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
        if i >= num_categories:
            break
        cat_dir = os.path.join(im_dir, directory)
        
        j = 0
        for fil in os.scandir(cat_dir):
            if j >= images_per_category:
                break
            if imghdr.what(fil) == 'ppm':
                image = cv2.imread(fil.path)
                images.append(cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT)))
                labels.append(i)
                j+= 1

    return images, labels

def load_testing_data(data_dir, num_categories, num_images = 10000):
    """
    Load image data from directory 'data_dir'.

    `data_dir` must have a sub-directory called 'Images', which in must have
    some number of .ppm image files (names of image files do not matter). 
    Files with different extensions will be ignored.  Inside of `data_dir`, 
    there should be a csv file `GT-final_test.csv` with the labels of the image data.

    The function returns a tuple `(images, labels)` which is a list of all the
    images in the data directory, where each image is formatted as a numpy ndarray
    with dimensions IMG_WIDTH x IMG_HEIGHT x 3.  `labels` is a list of integer labels,
    representing the categories for each of the corresponding `images`.
    """

    images, labels = [], []
    im_dir = os.path.join(data_dir, 'Images')

    with open(os.path.join(data_dir, "GT-final_test.csv")) as f:
        reader = csv.DictReader(f, delimiter = ';')
        j = 0
        for row in reader:
            label = int(row['ClassId'])
            if label >= num_categories:
                continue
            else:
                image = cv2.imread(os.path.join(im_dir, row['Filename']))
                images.append(cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT)))
                labels.append(label)
                j+= 1
            if j >= num_images:
                break

    return images, labels

def get_model(saved_model, num_categories = None):
    """
    Returns a compiled convolutional neural network model. The 'input_shape'
    of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.

    The output layer will have `num_categories` neurons; one for each category.
    """
    
    if saved_model:
        model = tf.keras.models.load_model(saved_model)
        print(f"Model loaded from {saved_model}")
        if num_categories and model.output.shape[1] != num_categories:
            raise ValueError("The number of categories does not agree with the output shape of the loaded model.")
        return model
    elif num_categories:
        return get_project_model(num_categories)
    else:
        return get_project_model(43)



def get_project_model(num_categories):
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
    model.add(tf.keras.layers.Dense(num_categories * 32, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))

    # Add hidden layer
    model.add(tf.keras.layers.Dense(num_categories * 16, activation="relu"))

    # Add output layer
    model.add(tf.keras.layers.Dense(num_categories, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


if __name__ == "__main__":
    main()
