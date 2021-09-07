# Overview

This project contains a convolutoinal neural network to classify images of German traffic signs.



```
$ python traffic.py gtsrb
Train on 15984 samples
Epoch 1/10
15984/15984 [==============================] - 10s 623us/sample - loss: 2.8565 - accuracy: 0.3022
Epoch 2/10
15984/15984 [==============================] - 8s 510us/sample - loss: 1.3484 - accuracy: 0.5951
Epoch 3/10
15984/15984 [==============================] - 8s 531us/sample - loss: 0.8283 - accuracy: 0.7494
Epoch 4/10
15984/15984 [==============================] - 12s 736us/sample - loss: 0.5758 - accuracy: 0.8270
Epoch 5/10
15984/15984 [==============================] - 12s 744us/sample - loss: 0.4241 - accuracy: 0.8725
Epoch 6/10
15984/15984 [==============================] - 10s 602us/sample - loss: 0.3391 - accuracy: 0.8956
Epoch 7/10
15984/15984 [==============================] - 10s 620us/sample - loss: 0.3102 - accuracy: 0.9103
Epoch 8/10
15984/15984 [==============================] - 11s 668us/sample - loss: 0.2747 - accuracy: 0.9207
Epoch 9/10
15984/15984 [==============================] - 10s 614us/sample - loss: 0.2208 - accuracy: 0.9362
Epoch 10/10
15984/15984 [==============================] - 8s 528us/sample - loss: 0.1961 - accuracy: 0.9418
10656/10656 - 2s - loss: 0.1392 - accuracy: 0.9606
```

## Dataset

[German Traffic Sign Recognition Dataset (GTSRB)](http://benchmark.ini.rub.de/index.php?section=gtsrb&subsection=about) is an image classification dataset.  The images are photos of traffic signs.  The images are classified into 43 classes.  The training set contains 39209 labeled images and the test set contains 12630 images.  Labels for the test set are published. 
## Model

As a first pass, we implement a traditioal CNN structured with layers as follows:
- a convolutional layer
- a max-pooling layer
- a flattening layer
- a dense layer with dropout (0.2)
- another dense layer
- an output layer.

## Performance/Evaluation

The baseline model achieves performance of $96-97\%$ accuracy when evaluated on the testing set with 5 minutes of training on a basic laptop.
