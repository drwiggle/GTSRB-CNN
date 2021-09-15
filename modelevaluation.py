import numpy as np
import tensorflow as tf

from classifier import load_testing_data, load_training_data, get_model

import matplotlib.pyplot as plt

args = {
    "load": "ProjectModel.h5",
    "num_images": 15000,
    "testing_data_directory": "gtsrb-testing",
    "training_data_directory": "gtsrb-training"
}

############################################################################

def load_rep_images(training_data_directory, num_categories):
    reps, _ = load_training_data(args["training_data_directory"], num_categories, 5)
    reps = [img for i, img in enumerate(reps) if i % 5 == 2]
    return reps
    
def find_incorrect_classifications(x_test, y_test, y_pred):
    """
    Compare predictions with actual classifications and make a list
     of incorrect predictions.  Return a list of images that are incorrectly
    classified and a list of their (incorrect) classifications.
    """

    bad_imgs, bad_img_preds = [], []
    for i, img in enumerate(x_test):
        if y_pred[i] != y_test[i]:
            bad_imgs.append(img)
            bad_img_preds.append(y_pred[i])

    return bad_imgs, bad_img_preds


def show_inc_class_imgs(bad_imgs, bad_img_preds, reps):
    """
    Show incorrectly classified images in a figure.

    Columns Alternate between 
    - incorrectly classified image, and
    - a representative sample of the way said image was classified
    
    """
    l = 2 * (1 + int(np.sqrt(len(bad_img_preds)))) # num cols
    f = plt.figure()
    for i, img in enumerate(bad_imgs):
        f.add_subplot(l, l, 2*i+1)
        plt.imshow(img)
        f.add_subplot(l, l, 2*i+2)
        plt.imshow(reps[bad_img_preds[i]])

    plt.show()

def hist_act_types_of_bad_imgs(y_test, y_pred, num_categories, show = True):
    """
    Makes a histogram showing the actual types of the misclassified
    images.  This will help us determine which image (types) are most
    commonly misclassified.

    Returns the bins from the histogram each entry in bins1 is a category,
    and the value of that entry is the number of misclassified images from
    that category.
    
    If show is set to False, the histogram is not displayed.
    """
    x1 = [y_test[i] for i in range(len(y_test)) if y_test[i]!=y_pred[i]]
    bins1 = [x1.count(i) for i in range(num_categories)]
    if show:
        plt.hist(x1, bins = num_categories)
        plt.xlabel('Image categories')
        plt.xticks(range(num_categories + 1), range(1,num_categories+2))
        plt.ylabel('Count')
        plt.title('Actual Types of the Misclassified Images')
        plt.show()
    return bins1

def hist_classifications_of_bad_imgs(y_test, y_pred, num_categories, show = False):
    """
    Makes a histogram showing the classifications of the misclassified
    images.  This will help us determine which how images are most
    commonly misclassified.

    Returns the bins from the histogram each entry in bins2 is a category,
    and the value of that entry is the number images that are incorrectly 
    classified in that category.
    
    If show is set to False, the histogram is not displayed.
    """

    x2 = [y_pred[i] for i in range(len(y_pred)) if y_test[i]!=y_pred[i]]
    bins2 = [x2.count(i) for i in range(num_categories)]
    if show:
        plt.hist(x2, bins = num_categories)
        plt.xlabel('Image categories')
        plt.xticks(range(num_categories+1), range(1,num_categories+2))
        plt.ylabel('Count')
        plt.title('Classifications of the Misclassified Images')
        plt.show()

    return bins2

def show_bad(bad_types, bad_bins, reps):
    """
    Show the bad types and bad bins in a figure.
    Use a representative sample from each category
    
    The first row displays the actual types of the most frequently
    misclassified images.  The second row shows the most frequent 
    misclassifications.
    """
    f = plt.figure()
    w = max(len(bad_bins), len(bad_types))
    for i, ty in enumerate(bad_types):
        f.add_subplot(2, w, i+1)
        plt.imshow(reps[ty])
        for i, bi in enumerate(bad_bins):
            f.add_subplot(2, w, w+i+1)
            plt.imshow(reps[bi])
    plt.show()

def bin_misclassified_images(y_test, y_pred, num_categories):
    """
    Returns a dict whose keys are image categories 
    (integers 0,1,...,num_categories - 1) and such that 
    res[i] is a list of length num_categories whose j-th entry is the number
    of images of type i that were classified as type j.

    This will let us break down the misclassifications by image type.
    """
    misclass_imgs_by_type = {i:[] for i in range(num_categories)}
    for act_cls, pred_cls in zip(y_test, y_pred):
        if act_cls != pred_cls:
            misclass_imgs_by_type[act_cls].append(pred_cls)
            
    bins = dict()
    for i in range(num_categories):
        bins[i] = [misclass_imgs_by_type[i].count(j) for j in range(num_categories)]

    return bins

def common_cls_of_bad_types(types, misclass_img_bins, proportion):
    """
    Given a list of image categories (types), for each category i (in types),
    determine the most common ways that images of type i are misclassified.

    Proportion determines how many types to return.

    Returns a dict whose keys are types and values are lists containing
    the most common ways for images of that type to be (mis)classificatied.
    """
    mixups = dict()
    for i in types:
        bins = misclass_img_bins[i]
        worst = max(bins)
        mixups[i] = [j for j,c in enumerate(bins) if c>= proportion * worst]

    return mixups

def show_common_mixups(mixups, reps):
    """
    Display the common mixups in a grid.  Each row is an image type i
    that is commonly miscategorized.  The first entry of the row is
    a representative image of type i.  The remaining entries are representative
    images for the most common ways that images of type i are miscategorized.
    """
    f = plt.figure()
    w = 2 + max([len(mixups[i]) for i in mixups])
    h = len(mixups)
    for i, k in enumerate(mixups):
        val = mixups[k]
        f.add_subplot(h, w, i * w + 1)
        plt.imshow(reps[k])

        for j, cls in enumerate(val):
            f.add_subplot(h, w, i * w + 3 + j)
            plt.imshow(reps[cls])

    plt.show()

############################################################################

# Load the desired model and determine number of categories
model = get_model(args["load"], None)
num_categories = model.output.shape[1]


# Load a representative image from each category
reps = load_rep_images(args["training_data_directory"], num_categories)


# Load images and labels for entire training set
testing_images, testing_labels = load_testing_data(
    args["testing_data_directory"],
    num_categories,
    args["num_images"]
)

# Convert loaded images (and labels) to numpy arrays and
# determine predictions for model on each testing image
x_test, y_test = np.array(testing_images), np.array(testing_labels)
y_pred = np.argmax(model.predict(x_test), axis = -1)


# Determine the incorrectly classified images (along with the way
# each was classified), then display in a figure
bad_imgs, bad_img_preds = find_incorrect_classifications(x_test, y_test, y_pred)

# Only run this if bad_imgs is small
# show_inc_class_imgs(bad_imgs, bad_img_preds, reps)


# Make histograms to analyze the misclassified images

# First we determine the actual type of each misclassified image
# This can be displayed in a histogram.
# Find the most frequently misclassified image types
bins1 = hist_act_types_of_bad_imgs(y_test, y_pred, num_categories, show = False)
worst = max(bins1)
bad_types =  [i for i,c in enumerate(bins1) if c > .75 * worst]

# Make histogram of the classifications of the misclassified images
# Then the most frequently misclassification categories
bins2 = hist_classifications_of_bad_imgs(y_test, y_pred, num_categories, show = False)
worst = max(bins2)
bad_bins =  [i for i,c in enumerate(bins2) if c> .75 * worst]


# display a representative sample from each category
# show_bad(bad_types, bad_bins, reps)


# Determine how the misclassified images are being classified
# Broken down by image type.
misclass_img_bins = bin_misclassified_images(y_test, y_pred, num_categories)

# Now analyze the bad_types (the image types that are most
# frequently misclassified) and see how these types are most often
# classified.

common_mixups = common_cls_of_bad_types(bad_types, misclass_img_bins, 0.4)

print(common_mixups)

show_common_mixups(common_mixups, reps)

