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
import numpy as np
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
X_train = X[:int(total*0.8),:,:,:]
Y_train = Y[:int(total*0.8)]
X_valid = X[int(total*0.8):,:,:,:]
Y_valid = Y[int(total*0.8):]
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
ful_layer = tflearn.fully_connected(avg_layer, 100, name='ful_layer')
softmax_layer = tflearn.activations.softmax(ful_layer)

# Regression
mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
net = tflearn.regression(softmax_layer, optimizer=mom,
                         loss='categorical_crossentropy')
model = tflearn.DNN(net)
model_without_softmax = tflearn.DNN(act_layer)

'''
# Training
model = tflearn.DNN(net, checkpoint_path='model_resnet_cifar100-N',
                    best_checkpoint_path='model_resnet_cifar100_best-N',
                    best_val_accuracy=70,
                    max_checkpoints=1, tensorboard_verbose=3,
                    clip_gradients=0.)
model.fit(X, Y, n_epoch=100, validation_set=(testX, testY),
          snapshot_epoch=False, snapshot_step=1000,
          show_metric=True, batch_size=64, shuffle=True,
          run_id='resnet_cifar100')
model.save("model-resnet-cifar100-110L-N-sean")
'''
################################
###### Secondary training ######
################################

### load pre-trained image model
model.load('model-resnet-cifar100-110L-N-sean')
print('Image model loaded')

# predict X by using pre-trained model
X_impred = np.zeros((total, 8, 8, 64)) # Output of image model with X being as its input
testX_impred = np.zeros((testX.shape[0], 8, 8, 64))
batch_size = 10
for i in range(int(total/batch_size)):
  X_impred[i*batch_size: (i+1)*batch_size] = model_without_softmax.predict(X[i*batch_size: (i+1)*batch_size])
for i in range(int(testX.shape[0]/batch_size)):
  testX_impred[i*batch_size: (i+1)*batch_size] = model_without_softmax.predict(testX[i*batch_size: (i+1)*batch_size])

X_impred_train = X_impred[:int(total*0.8),:,:,:]
X_impred_valid = X_impred[int(total*0.8):,:,:,:]
testX_impred = testX_impred
print('X_predicted value loaded')
np.save('X_impred_train', X_impred_train)
np.save('X_impred_valid', X_impred_valid)
np.save('testX_impred', testX_impred)
print()
### load pre-trained word embedding
with open('words500.vocab', 'r') as file:
  vocab = file.read()
  vocab = vocab.split('\n')
embedding = np.load('words500.npy')
embedding_matrix = np.matrix(embedding)
dictionary = dict(zip(vocab, embedding))
print('Word embedding dictionary created')

# load labels of cifar100
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
labels_dict = unpickle('cifar-100-python/meta')
fine_label_names = labels_dict[b'fine_label_names']
coarse_label_names = labels_dict[b'coarse_label_names']

# process images' labels and convert to vectors
Y_train_label_position = np.argmax(Y_train, axis=1)
Y_valid_label_position = np.argmax(Y_valid, axis=1)
testY_label_position = np.argmax(testY, axis=1)

Y_train_label = []
Y_valid_label = []
testY_label = []

Y_train_vec = []
Y_valid_vec = []
testY_vec = []

for position in Y_train_label_position:
  this_name = fine_label_names[position].decode('utf-8')
  Y_train_label.append(this_name)
  multi_words = False
  for i in range(len(this_name)):
    if this_name[i] == '_':
      multi_words = True
      break
  if multi_words:
    words_list = this_name.split('_')
    sum_vec = np.zeros(500)
    for word in words_list:
      sum_vec = np.add(sum_vec, dictionary[word])
    Y_train_vec.append(sum_vec)
  else:
    words_list = this_name
    Y_train_vec.append(dictionary[this_name])

print('Training Set label loaded')

for position in Y_valid_label_position:
  this_name = fine_label_names[position].decode('utf-8')
  Y_valid_label.append(this_name)
  multi_words = False
  for i in range(len(this_name)):
    if this_name[i] == '_':
      multi_words = True
      break
  if multi_words:
    words_list = this_name.split('_')
    sum_vec = np.zeros(500)
    for word in words_list:
      sum_vec = np.add(sum_vec, dictionary[word])
    Y_valid_vec.append(sum_vec)
  else:
    words_list = this_name
    Y_valid_vec.append(dictionary[this_name])

print('Validation Set label loaded')

for position in testY_label_position:
  this_name = fine_label_names[position].decode('utf-8')
  testY_label.append(this_name)
  multi_words = False
  for i in range(len(this_name)):
    if this_name[i] == '_':
      multi_words = True
      break
  if multi_words:
    words_list = this_name.split('_')
    sum_vec = np.zeros(500)
    for word in words_list:
      sum_vec = np.add(sum_vec, dictionary[word])
    testY_vec.append(sum_vec)
  else:
    words_list = this_name
    testY_vec.append(dictionary[this_name])

print('Test Set label loaded')

np.save('Y_train_vec', Y_train_vec)
np.save('Y_valid_vec', Y_valid_vec)
np.save('testY_vec', testY_vec)

'''
### Build DEViSE model
input_placeholder_II = tf.placeholder(tf.float32, shape=(None, 8, 8, 64))
input_layer_II = tflearn.input_data(shape=[None, 8, 8, 64], placeholder=input_placeholder_II, name='input_layer_II')
flat_layer_II = tflearn.flatten(input_layer_II, name='flat_layer_II')
act_layer_II = tflearn.leaky_relu(flat_layer_II, name = 'act_layer_II')
ful_layer_II = tflearn.fully_connected(act_layer_II, 500, name='ful_layer_II')


# Regression
trueY_vec = tf.placeholder(tf.float32, shape=(None, 500))
loss_II = tf.losses.hinge_loss(trueY_vec, ful_layer_II)
mom_II = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
net_II = tflearn.regression(ful_layer_II, optimizer=mom_II,
                         loss=loss_II, name='regression_II')
model_II = tflearn.DNN(net_II)

# Training
model_II = tflearn.DNN(net_II, checkpoint_path='model_DEViSE-N',
                    best_checkpoint_path='model_DEViSE_best-N',
                    best_val_accuracy=70,
                    max_checkpoints=1, tensorboard_verbose=3,
                    clip_gradients=0.)
model_II.fit({input_placeholder_II: X_impred_train}, {trueY_vec: Y_train_vec}, n_epoch=75, 
          validation_set=({input_placeholder_II: X_impred_valid}, {trueY_vec: Y_valid_vec}),
          snapshot_epoch=False, snapshot_step=1000,
          show_metric=True, batch_size=64, shuffle=True,
          run_id='DEViSE')
model_II.save("model-DEViSE-110L-N-sean")

"""
all_vars = tf.train.list_variables('./model_resnet_cifar100-N-57000')
print('all', all_vars)
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
'''