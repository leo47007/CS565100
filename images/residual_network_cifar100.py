# -*- coding: utf-8 -*-

""" Deep Residual Network.

Applying a Deep Residual Network to CIFAR-10 Dataset classification task.

References:
    - K. He, X. Zhang, S. Ren, and J. Sun. Deep Residual Learning for Image
      Recognition, 2015.
    - Learning Multiple Layers of Features from Tiny Images, A. Krizhevsky, 2009.

Links:
    - [Deep Residual Network](http://arxiv.org/pdf/1512.03385.pdf)
    - [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import tflearn

# Residual blocks
# 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
n = 18 

# Data loading
from tflearn.datasets import cifar100
(X, Y), (testX, testY) = cifar100.load_data()
# divided into validation set, training set
total = X.shape[0]
X_train = X[:(total*0.8),:,:,:]
Y_train = Y[:(total*0.8),:,:,:]
X_valid = X[(total*0.2):,:,:,:]
Y_valid = Y[(total*0.2):,:,:,:]
# one-hot encoded
Y_train = tflearn.data_utils.to_categorical(Y_train, 100)
Y_valid = tflearn.data_utils.to_categorical(Y_valid, 100)
testY = tflearn.data_utils.to_categorical(testY, 100)

# Real-time data preprocessing
img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center(per_channel=True)

# Real-time data augmentation
img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_crop([32, 32], padding=4)

# Building Residual Network
input_layer = tflearn.input_data(shape=[None, 32, 32, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug, name='input_layer')
conv_layer = tflearn.conv_2d(input_layer, 16, 3, regularizer='L2', weight_decay=0.0001, name='conv_layer')
block1 = tflearn.residual_block(conv_layer, n, 16, name='block1')
block2 = tflearn.residual_block(block1, 1, 32, downsample=True, name='block2')
block3 = tflearn.residual_block(block2, n-1, 32, name='block3')
block4 = tflearn.residual_block(block3, 1, 64, downsample=True, name='block4')
block5 = tflearn.residual_block(block4, n-1, 64, name='block5')
nor_layer = tflearn.batch_normalization(block5, name='nor_layer')
act_layer = tflearn.activations.leaky_relu(nor_layer, alpha=0.1, name='act_layer')
avg_layer = tflearn.global_avg_pool(act_layer, name='avg_layer')

# Regression
ful_layer = tflearn.fully_connected(avg_layer, 100, activation='softmax', name='ful_layer')
mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
net = tflearn.regression(ful_layer, optimizer=mom,
                         loss='categorical_crossentropy')
model = tflearn.DNN(net)


# Training
model = tflearn.DNN(net, checkpoint_path='model_resnet_cifar100-N',
                    best_checkpoint_path='model_resnet_cifar100_best-N',
                    best_val_accuracy=70,
                    max_checkpoints=1, tensorboard_verbose=3,
                    clip_gradients=0.)

model.fit(X_train, Y_, n_epoch=100, validation_set=(testX, testY),
          snapshot_epoch=False, snapshot_step=1000,
          show_metric=True, batch_size=64, shuffle=True,
          run_id='resnet_cifar100')

model.save("model-resnet-cifar100-110L-N")
"""
# Load Model
model.load('./model_resnet_cifar100-N-57000')
all_vars = tf.train.list_variables('./model_resnet_cifar100-N-57000')
#print('all', all_vars)
#load_1 = tflearn.variables.get_layer_variables_by_name('block1')
#load_1 = tf.train.load_variable('./model_resnet_cifar100-N-57000','block1')
print('value', load_1)
#load_2 = tf.train.load_variable('./model_resnet_cifar100-N-57000',all_vars[9][0])
#print('value', load_2, 'shape=', load_2.shape)

score = model.evaluate(testX, testY)
print("Score=", score)
prediction = model.predict(testX)
print("prediction:", prediction)
"""

