import numpy as np
import cv2
import matplotlib.pyplot as plt

from classifier import load_testing_data, load_training_data, get_model
from modelevaluation import load_rep_images


args = {
    "load": "ProjectModel.h5",
    "num_images": 15000,
    "testing_data_directory": "gtsrb-testing",
    "training_data_directory": "gtsrb-training"
}

############################################################################
class ImageCollection:
    def __init__(self, orig):
        self.imdict = dict()
        self.imdict['orig'] = orig
        self.CLAHE_imgs = dict()

    def add_image(self, identifier, img):
        self.imdict[identifier] = img

    def get_image(self, identifier):
        return self.imdict[identifier]

    def add_CLAHE_image(self, clip_limit, grid_size, identifier, img):
        if clip_limit not in self.CLAHE_imgs:
            self.CLAHE_imgs[clip_limit] = dict()
        if grid_size not in self.CLAHE_imgs[clip_limit]:
            self.CLAHE_imgs[clip_limit][grid_size] = dict()
        self.CLAHE_imgs[clip_limit][grid_size][identifier] = img
        return None

    def get_CLAHE_image(self, clip_limit, grid_size, identifier):
        return self.CLAHE_imgs[clip_limit][grid_size][identifier]
        
def display_imgs(badsamples, headers):
    """
    Displays a grid of images: all images in a single row are enhanced versions
    of the original.  Different rows correspond to different image categories.

   `badsamples` is a dict whose keys are image categories, and whose values are
    themselves dicts, with each key being a string and each value an image.  For
    example `badsamples[i]` may have keys {"orig", "gray", "gray_hist", "eq_gray",
    "gray_clahe", etc.}

    `headers` is a list of strings describing which images from `badsamples[i]`
    to show.  Allowable strings are:
    - "orig": the original image
    - "gray": the grayscale image
    - "gray_hist": a histogram showing the intensities in the grayscale image
    - "eq_gray": an equalized grayscale image made using cv2.equalizeHist
    - "gray_clahe": CLAHE applied to the original grayscale
    - "gray_eq_clahe": CLAHE applied to eq_gray
    """
    f = plt.figure()
    w = len(headers)
    h = 1 + len(badsamples.keys())
    for j, im_type in enumerate(headers):
        f.add_subplot(h, w, j+1)
        plt.text(
            0.5,
            0.5,
            im_type,
            fontsize = 20,
            horizontalalignment='center',
            verticalalignment='center'
        )
        plt.axis('off')
        
    for i, img_coll in enumerate(badsamples.values()):
        for j, im_type in enumerate(headers):
            f.add_subplot(h, w, (i+1)*w + 1 + j)
            plt.imshow(img_coll.get_image(im_type))
            plt.axis('off')

    plt.show()
    return None

def display_CLAHE_imgs(img_coll, ID):
    """
    Displays a grid of images of type ID which have had various CLAHE objects applied to them
    ID can have any of the values: 'lab', 'gray', 'gray_eq', 'lab_eq'
    """
    CLAHE_imgs = img_coll.CLAHE_imgs
    clip_limits = list(CLAHE_imgs.keys())
    grid_sizes = list(CLAHE_imgs[clip_limits[0]].keys())
    
    f = plt.figure()
    w = 2 + len(clip_limits)
    h = 1 + len(grid_sizes)
    f.add_subplot(h, w, 1)
    plt.text(
        0.5,
        0.5,
        ID,
        fontsize = 20,
        horizontalalignment='center',
        verticalalignment='center'
    )
    plt.axis('off')
    f.add_subplot(h, w, 2)
    plt.text(
        0.5,
        0.5,
        'clip ->',
        fontsize = 20,
        horizontalalignment='center',
        verticalalignment='center'
    )
    plt.axis('off')
    
    for j, clip in enumerate(clip_limits):
        f.add_subplot(h, w, j+3)
        plt.text(
            0.5,
            0.5,
            str(clip),
            fontsize = 20,
            horizontalalignment='center',
            verticalalignment='center'
        )
        plt.axis('off')
        
    for i, grid in enumerate(grid_sizes):
        f.add_subplot(h, w, (i+1)*w + 1)
        plt.imshow(img_coll.get_image(ID))
        f.add_subplot(h, w, (i+1)*w + 2)
        plt.text(
            0.5,
            0.5,
            str(grid),
            fontsize = 20,
            horizontalalignment='center',
            verticalalignment='center'
        )
        plt.axis('off')

        for j, clip in enumerate(clip_limits):
            f.add_subplot(h, w, (i+1)*w + 3 + j)
            plt.imshow(img_coll.get_CLAHE_image(clip, grid, ID))
            plt.axis('off')

    plt.show()
    return None
############################################################################

# Load a representative image from each category
reps = load_rep_images(args["training_data_directory"], num_categories)

# initialize dict with bad image samples
badsamples = dict()
for i in [2, 3, 8, 11, 18, 26, 30]:
    badsamples[i] = ImageCollection(reps[i])

# make grayscale images and lab images
for i in badsamples:
    gray_img = cv2.cvtColor(badsamples[i].get_image('orig'), cv2.COLOR_BGR2GRAY)
    badsamples[i].add_image('gray', gray_img)
    lab_img = cv2.cvtColor(badsamples[i].get_image('orig'), cv2.COLOR_BGR2LAB)
    badsamples[i].add_image('lab', lab_img)

# analyze grayscale and color images using a histogram and equalize
for i in badsamples:
    hist = cv2.calcHist(badsamples[i].get_image('gray'), [0], None, [256], [0,256])
    badsamples[i].add_image('gray_hist', hist)
    gray_eq = cv2.equalizeHist(badsamples[i].get_image('gray'))
    badsamples[i].add_image('gray_eq', gray_eq)
    
    lab_img = badsamples[i].get_image('lab')
    lab_planes = cv2.split(lab_img)
    lab_planes[0] = cv2.equalizeHist(lab_planes[0])
    lab_img_eq = cv2.merge(lab_planes)
    badsamples[i].add_image('lab_eq', lab_img_eq)
    
# Create CLAHE objects for all cliplimits and grid_sizes
# apply CLAHE to grayscale (both equalized and not)
clahe = dict()
clip_limits = [5, 10, 20, 30, 40]
grid_sizes = [2, 4, 8, 16, 32]

for clip in clip_limits:
    clahe[clip] = dict()
    for tile in grid_sizes:
        clahe[clip][tile] = cv2.createCLAHE(
            clipLimit = clip,
            tileGridSize = (tile, tile)
        )

for clip in clip_limits:
    for tile in grid_sizes:
        cl_obj = clahe[clip][tile]
        for i in badsamples:
            # Apply CLAHE to gray image and gray_eq image add to img_coll
            for ID in ['gray', 'gray_eq']:

                img = cl_obj.apply(badsamples[i].get_image(ID))
                badsamples[i].add_CLAHE_image(clip, tile, ID, img)

            for ID in ['lab', 'lab_eq']:
                img = badsamples[i].get_image(ID)
                lab_planes = cv2.split(img)
                lab_planes[0] = cl_obj.apply(lab_planes[0])
                img = cv2.merge(lab_planes)
                badsamples[i].add_CLAHE_image(clip, tile, ID, img)

            

# display_imgs(badsamples, ['orig', 'gray', 'gray_eq', 'lab', 'lab_eq'])
display_CLAHE_imgs(badsamples[18], 'gray')
