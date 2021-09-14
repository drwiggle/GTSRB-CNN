import numpy as np
import tensorflow as tf

from classifier import load_testing_data, load_training_data, get_model

import matplotlib.pyplot as plt

args = {
    "load": "ProjectModel.h5",
    "num_images": 50,
    "testing_data_directory": "gtsrb-testing",
    "training_data_directory": "gtsrb-training"
}

model = get_model(args["load"], None)
num_categories = model.output.shape[1]

# Load a representative image from each category
reps, _ = load_training_data(args["training_data_directory"], num_categories, 1)

    # Load images and labels for entire training set
testing_images, testing_labels = load_testing_data(
    args["testing_data_directory"],
    num_categories,
    args["num_images"]
)

# Convert loaded images (and labels) to numpy arrays
x_test, y_test = np.array(testing_images), np.array(testing_labels)

# Determine predictions for model on each testing image
y_pred = np.argmax(model.predict(x_test), axis = -1)

# Compare predictions with actual classifications and make a list
# of incorrect predictions

bad_imgs, bad_img_preds = [], []
for i, img in enumerate(x_test):
    if y_pred[i] != y_test[i]:
        bad_imgs.append(img)
        bad_img_preds.append(y_pred[i])

# Show incorrectly classified images in a figure
# Columns Alternate (incorr. class, classification sample, ...)
l = 2 * (1 + int(np.sqrt(len(incorrect_preds)))) # num cols
f = plt.figure()
for i, img in enumerate(bad_imgs):
    f.add_subplot(l, l, 2*i+1)
    plt.imshow(img)
    f.add_subplot(l, l, 2*i+2)
    plt.imshow(reps[bad_img_preds[i]])

plt.show()
