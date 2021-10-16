import os

import numpy as np
import cv2

import classifier
from sklearn.model_selection import train_test_split

from modelevaluation import load_rep_images

import matplotlib.pyplot as plt

# set constants
args = {
    "images_per_category": 10000,
    "num_categories": 43,
    "testing_data_directory": "gtsrb-testing",
    "training_data_directory": "gtsrb-training",
    "epochs": 10
}

# set filenames for training and testing datasets
sm_filenames = {
    "x_train": "smx_train",
    "y_train": "smy_train",
    "x_test": "smx_test",
    "y_test": "smy_test"
}
lg_filenames = {
    "x_train": "x_train",
    "y_train": "y_train",
    "x_test": "x_test",
    "y_test": "y_test"
}
train_test_filenames = lg_filenames

# Set constants for CLAHE parameter ranges
grid_sizes = [2, 4, 8]
clip_limits = [5, 10, 20, 30, 40]
ids = ['gray', 'gray_eq', 'lab', 'lab_eq', 'orig']

TEST_SIZE = 0.20              # Allows user to modify proportion of data set

############################################################################
def load_split_save_images(
        training_dir,
        num_categories,
        images_per_category,
        tt_filenames):
    """
    Takes as input the training directory, the number of categories, number of
    images per category, and a dictionary tt_filenames specifying how to label
    the saved training/testing sets.
    
    This method will load the specified images from the training directory,
    then split them into training and testing sets, then save the training
    and testing sets to files in a directory called 'presplitimages' with
    filenames specified by the last input parameter.
    
    This method should be run ONE TIME for a specified set of models.  All models
    can then be trained and tested on the same data.
    """
    # load training data
    images, labels = load_training_data(
        training_dir,
        num_categories,
        images_per_category
    )

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Save training and testing sets to files
    np.save(f"{os.path.join('presplitimages',tt_filenames['x_train'])}.npy", x_train)
    np.save(f"{os.path.join('presplitimages',tt_filenames['y_train'])}.npy", y_train)
    np.save(f"{os.path.join('presplitimages',tt_filenames['x_test'])}.npy", x_test)
    np.save(f"{os.path.join('presplitimages',tt_filenames['y_test'])}.npy", y_test)

    return x_train, x_test, y_train, y_test

def load_presplit_images(tt_filenames):
    """
    Loads a saved set of testing and training images from the directory
    'presplitimages/'.  This is so we
    can use the same training/testing sets on many models.
    """
    x_train = np.load(f"{os.path.join('presplitimages',tt_filenames['x_train'])}.npy")
    x_test = np.load(f"{os.path.join('presplitimages',tt_filenames['x_test'])}.npy")
    y_train = np.load(f"{os.path.join('presplitimages',tt_filenames['y_train'])}.npy")
    y_test = np.load(f"{os.path.join('presplitimages',tt_filenames['y_test'])}.npy")
    return x_train, x_test, y_train, y_test

    
def get_mod_images(x_train_orig, x_test_orig, im_type, cl = None):
    """
    Returns two lists of modified images, modified according to
    im_type (which can be 'orig', 'gray', 'gray_eq', 'lab', or 'lab_eq')
    and cl which can specify the parameters for a CLAHE filter.

    cl, if specified should have the form (grid_size, clip_limit).
    """
    x_train, x_test = [], []

    # If im_type is orig, do not modify
    if im_type == 'orig':
        return x_train_orig, x_test_orig
    
    # Set the filter to be applied according to im_type
    if im_type[:4] == 'gray':
        filt = cv2.COLOR_BGR2GRAY
    elif im_type[:3] == 'lab':
        filt = cv2.COLOR_BGR2LAB

    # Apply the filter
    for img in x_train_orig:
        x_train.append(cv2.cvtColor(img, filt))
    for img in x_test_orig:
        x_test.append(cv2.cvtColor(img, filt))

    # If images should be equalized, apply histogram
    # appropriately
    if im_type == "gray_eq":
        x_train = [cv2.equalizeHist(img) for img in x_train]
        x_test =  [cv2.equalizeHist(img) for img in x_test]

    if im_type == "lab_eq":
        for i in range(len(x_train)):
            lab_planes = cv2.split(x_train[i])
            lab_planes[0] = cv2.equalizeHist(lab_planes[0])
            x_train[i] = cv2.merge(lab_planes)
        for i in range(len(x_test)):
            lab_planes = cv2.split(x_test[i])
            lab_planes[0] = cv2.equalizeHist(lab_planes[0])
            x_test[i] = cv2.merge(lab_planes)

    # Convert to numpy ndarrays, then reshape if necessary
    x_train, x_test = np.array(x_train), np.array(x_test)

    # grayscale images will now have the incorrect shape
    # as ndarrays, so reshape them
    dim = len(x_train[0].shape)
    if dim == 2:
        width, height = x_train.shape[1], x_train.shape[2]
        x_train = x_train.reshape(x_train.shape[0],width, height, 1)
        x_test = x_test.reshape(x_test.shape[0],width, height, 1)

    # Return modified images, or apply CLAHE filter as appropriate
    if not cl:
        return x_train, x_test
    else:
        grid_size, clip_limit = cl[0], cl[1]
        cl_obj = cv2.createCLAHE(
            clipLimit = clip_limit,
            tileGridSize = (grid_size, grid_size)
        )
        if im_type[:4] == "gray":
            for coll in [x_train, x_test]:
                for i in range(len(coll)):
                    img = coll[i]
                    mod_img = cl_obj.apply(img)
                    mod_img = mod_img.reshape(mod_img.shape[0], mod_img.shape[1], 1)
                    coll[i] = mod_img
        elif im_type[:3] == "lab":
            for coll in [x_train, x_test]:
                for i in range(len(coll)):
                    img = coll[i]
                    lab_planes = cv2.split(img)
                    lab_planes[0] = cl_obj.apply(lab_planes[0])
                    mod_img = cv2.merge(lab_planes)
                    coll[i] = mod_img
    
    return x_train, x_test

def make_train_evaluate_model(
        x_train_orig,
        x_test_orig,
        y_train,
        y_test,
        im_type,
        cl = None):
    """
    Makes a model, modifies the training/testing images according to im_type, then
    trains and evaluates the model on the specified training/testing sets.  

    im_type can be any of 'orig', 'gray', 'lab', 'gray_eq', 'lab_eq'

    Returns a dictionary whose keys are {'training_loss', 'training_acc',
    'testing_loss', 'testing_acc'}
    """
    ch = 1 if im_type[:4] == 'gray' else 3
    x_train, x_test = get_mod_images(x_train_orig, x_test_orig, im_type, cl)
    model = get_model(None, num_categories = args["num_categories"], channels = ch)
    model.fit(x_train, y_train, epochs = args['epochs'], verbose = 2)
    print(f"done training {im_type} model, with {cl if cl else 'no'} CLAHE filter")
    model_res = dict()
    res = model.evaluate(x_train, y_train, verbose = 0)
    model_res['training_loss'] = res[0]
    model_res['training_acc'] = res[1]
    res = model.evaluate(x_test, y_test, verbose = 0)
    model_res['testing_loss'] = res[0]
    model_res['testing_acc'] = res[1]
    return model_res

def show_hist_results(results, ids):
    """
    Shows a histogram of the results from the initial comparison of models
    trained on the same set of images, each with a different filter applied.
    The filters are {'orig', 'gray', 'gray_eq', 'lab', 'lab_eq'}
    """
    labels = ids
    tr_acc = [results[name]['training_acc'] for name in labels]
    te_acc = [results[name]['testing_acc'] for name in labels]

    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x-width/2, tr_acc, width, label = "tr acc.")
    rects2 = ax.bar(x+width/2, te_acc, width, label = "te acc.")
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by image type on training/testing sets')
    ax.set_xticks(x)
    ax.set_xticklabels([label[1:] for label in labels])
    ax.legend()
    
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.ylim(.8, 1)
    
    plt.show()

def show_CLAHE_3d_bar(results, im_type, cutoff = .8):
    """
    Makes a 3d bar plot for the results of the models trained on image
    sets with CLAHE filters applied to `im_type` images.  `im_type` can
    be {gray, gray_eq, lab, lab_eq}.

    Cutoff specifies a lower cutoff for the z-axis.
    """
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # arange x and y values
    _x = np.arange(5)
    _y = np.arange(3)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    xtoCL = {a:clip for a,clip in zip(_x, clip_limits)}
    ytoGS = {b:size for b,size in zip(_y, grid_sizes)}

    
    top1 = [results[(im_type, ytoGS[b], xtoCL[a])]['training_acc'] - cutoff
            for a,b in zip(x,y)]
    top2 = [results[(im_type, ytoGS[b], xtoCL[a])]['testing_acc'] - cutoff
            for a,b in zip(x,y)]
    bottom = [cutoff for a in x]
    width = depth = 1

    ax1.bar3d(x, y, bottom, width, depth, top1, shade=True)
    ax1.set_title('Training Accuracy')
    ax1.set_xlabel('Clip Limit')
    ax1.set_xticks(_x)
    ax1.set_xticklabels(clip_limits)
    ax1.set_yticks(_y)
    ax1.set_yticklabels(grid_sizes)

    ax1.set_ylabel('Grid Size')
    ax1.set_zlim(cutoff,1)
    
    ax2.bar3d(x, y, bottom, width, depth, top2, shade=True)
    ax2.set_title('Testing Accuracy')
    ax2.set_xlabel('Clip Limit')
    ax1.set_xticks(_x)
    ax1.set_xticklabels(clip_limits)
    ax1.set_yticks(_y)
    ax1.set_yticklabels(grid_sizes)
    ax2.set_ylabel('Grid Size')
    ax2.set_zlim(cutoff,1)
    
    plt.show()


def compare_model_avgs(results, dim, vals):
    """
    Given an axis, `dim`, across which to compare (should be one of {'im_type',
    'clip_limit', 'grid_size'), and a list of values within that axis, this
    function returns a pair of dictionaries, one for testing results and
    another for training results.  Each of the dictionaries has `vals` as its
    keys.  The value of dict[key] for any given key is the average performance
    of all the models trained on that type of image (with various CLAHE filters).
    """
    dim_dict = {'im_type':0, 'grid_size': 1, 'clip_limit': 2}
    
    teres = {val: [] for val in vals}
    trres = {val: [] for val in vals}
    for key, val in results.items():
        if key[dim_dict[dim]] in vals:
            teres[key[dim_dict[dim]]].append(val['testing_acc'])
            trres[key[dim_dict[dim]]].append(val['training_acc'])
    for thing in vals:
        teres[thing] = sum(teres[thing])/len(teres[thing])
        trres[thing] = sum(trres[thing])/len(trres[thing])
    return teres, trres

############################################################################

# Load images, train_test_split them

# x_train, x_test, y_train, y_test = load_split_save_images(
#     args["training_data_directory"],
#     args["num_categories"],
#     args["images_per_category"],
#     train_test_filenames
# )

# load saved training/testing sets
# x_train_orig, x_test_orig, y_train_orig, y_test_orig = load_presplit_images(
#     train_test_filenames)
# y_train, y_test = y_train_orig, y_test_orig

# We need a bunch of models, one for each triple (clip, grid, im_type)
# For each triple, we will modify the training and testing sets
# to conform to the image type, then make and train a model, then evaluate
# the model.  Finally, we save the evaluation data to a dictionary.
# The dictionary is saved to a file.

# Make, train, and evaluate models without CLAHE filters applied
# results = dict()

# for im_type in ids:
#     print(f"model {im_type}")
#     results[im_type] = make_train_evaluate_model(
#         x_train_orig,
#         x_test_orig,
#         y_train,
#         y_test,
#         im_type
#     )

# for im_type in ids[:-1]:
#     for grid_size in grid_sizes:
#         for clip_limit in clip_limits:
#             print(f"Model {im_type} with CLAHE filter: clip = {clip_limit}, grid = {grid_size}")
#             results[(im_type, grid_size, clip_limit)] = make_train_evaluate_model(
#                 x_train_orig,
#                 x_test_orig,
#                 y_train,
#                 y_test,
#                 im_type,
#                 (grid_size, clip_limit)
#             )

# Show histogram of results for model accuracy on models without
# CLAHE filters
show_hist_results(results, ids)

# Show a 3d bar plot comparing performance of all CLAHE operators applied within a
# specified image type
show_CLAHE_3d_bar(results, 'lab_eq')
show_CLAHE_3d_bar(results, 'gray')

# Based on this analysis, the model that performs optimally does so on equalized lab images and uses a CLAHE filter with grid_size = 8 and clip_limit = 10.
