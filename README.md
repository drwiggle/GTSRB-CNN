# Overview

The purpose of this project is to develop a convolutional neural network to classify images of German traffic signs.

```
$ python classifier.py "gtsrb-training" -td "gtsrb-testing"
Train on 39,252 samples
Epoch 1/10
39209/39209 [==============================] - 10s 623us/sample - loss: 2.8565 - accuracy: 0.3022
Epoch 2/10
39209/39209 [==============================] - 8s 510us/sample - loss: 1.3484 - accuracy: 0.5951
Epoch 3/10
39209/39209 [==============================] - 8s 531us/sample - loss: 0.8283 - accuracy: 0.7494
Epoch 4/10
39209/39209 [==============================] - 12s 736us/sample - loss: 0.5758 - accuracy: 0.8270
Epoch 5/10
39209/39209 [==============================] - 12s 744us/sample - loss: 0.4241 - accuracy: 0.8725
Epoch 6/10
39209/39209 [==============================] - 10s 602us/sample - loss: 0.3391 - accuracy: 0.8956
Epoch 7/10
39209/39209 [==============================] - 10s 620us/sample - loss: 0.3102 - accuracy: 0.9103
Epoch 8/10
39209/39209 [==============================] - 11s 668us/sample - loss: 0.2747 - accuracy: 0.9207
Epoch 9/10
39209/39209 [==============================] - 10s 614us/sample - loss: 0.2208 - accuracy: 0.9362
Epoch 10/10
39209/39209 [==============================] - 8s 528us/sample - loss: 0.1961 - accuracy: 0.9418
12630/12630 - 2s - loss: 0.1392 - accuracy: 0.9788
```

## Dataset

[German Traffic Sign Recognition Dataset (GTSRB)](http://benchmark.ini.rub.de/index.php?section=gtsrb&subsection=about) is an image classification dataset.  The images are photos of traffic signs; there are 43 classes of signs.  The training set contains 39209 labeled images and the test set contains 12630 images.  Labels for the test set are published.  Some sample images are shown below.

![Samples from GTSRB Dataset](https://github.com/drwiggle/GTSRB-CNN/sampleimgs.png)

## Model

As a first pass, we implement a traditional CNN structured with layers as follows:
- a convolutional layer
- a max-pooling layer
- a flattening layer
- a dense layer with dropout (0.2)
- another dense layer
- an output layer.

## Baseline Performance/Evaluation

The baseline model achieves performance of $$ 94-95\% $$ accuracy when evaluated on the testing set with 5 minutes of training on a basic laptop.

## First Improvements
As a second step, we investigate (in `misclassifiedimgs.py`) the most commonly misclassified images (and image types).  We find several "bad image types" (those which are most commonly misclassified).  Samples of the bad image types are shown here:


A close look at the images leads us to consider applying image enhancement techniques, like grayscale maps, histogram equalization, and CLAHE filters (Contrast Limited Adaptive Histogram Equalization).  This is explored in `visimgenhancement.py`, where we visually inspect how a variety of filters make "bad image types" more easily distinguishable (to the human eye).

Next, in `imgenhmodeltesting.py` we systematically apply to a single training/testing set the various image enhancement techniques previously explored.  We train (and evaluate) a model on the fixed image set (with various enhancement techniques applied).  We record the performance of each model with respect to its accuracy on both training and testing sets.  By comparing these performances (see images below), we conclude that the most effective technique is to convert the images to LAB; then apply a global histogram equalization to the (L)ightness parameter; then apply a CLAHE filter with `clip_limit = 10` and `tile_grid_size = 8`. 

With more computing power, it might be worthwhile to perform a statistical analysis of the `clip_limit` and `tile_grid_size` parameters that were explored to ensure this choice is factually based on a statistically significant different in the performance between models, and not due to random variation because of the particular training/testing set that was chosen.

## Performance Revisited
By incorporating contrast equalization and a CLAHE filter to `classifier.py`, the performance of our model improves to $96-97\%$ accuracy on the testing set and approximately $98\%$ accuracy on the training set.
