from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow as tf
import os
import tflearn
import csv

#################
### Load Data ###
#################

# load cyfar100 data
from tflearn.datasets import cifar100
(X, Y), (testX, testY) = cifar100.load_data()
X = np.asarray(X)
Y = np.asarray(Y)
testX = np.asarray(testX)
testY = np.asarray(testY)

# divided into validation set, training set
total = X.shape[0]
'''
X_train = X[:int(total*0.8),:,:,:]
Y_train = Y[:int(total*0.8)]
X_valid = X[int(total*0.8):,:,:,:]
Y_valid = Y[int(total*0.8):]

'''

# Load the predicted tensors and vector representations of label 
# from Residual Network Model
'''
X_impred_train = np.load('X_impred_train.npy')      # X train for DEViSE.py, which is a predicted tensor from Resnet
X_impred_valid = np.load('X_impred_valid.npy')      # X valid for DEViSE.py, which is a predicted tensor from Resnet
'''
X_impred = np.load('X_impred.npy')			# 12/29 modified
testX_impred = np.load('testX_impred.npy')          # X test for DEViSE.py, which is a predicted tensor from Resnet
'''
Y_train_vec = np.load('Y_train_vec.npy')            # Y train for DEViSE.py, which is a vector representation of label of dataset
Y_valid_vec = np.load('Y_valid_vec.npy')            # Y valid for DEViSE.py, which is a vector representation of label of dataset
'''
Y_vec = np.load('Y_vec.npy')				# 12/29 modified
testY_vec = np.load('testY_vec.npy')                # Y test for DEViSE.py, which is a vector representation of label of dataset
fine_label = np.load('fine_label_vec.npy').astype(np.float32)   # Vector representation of fine label of dataset
num_label = fine_label.shape[0]



### Environment settings###
# Setting Parameters and the path for TensorBoard
logs_path = 'TensorBoard/'
n_features = 64 # number of image pixels 
batch_size = 128 # Size of a mini-batch
epoch_num = 5000 # Epochs to train


# Launching InteractiveSession
sess = tf.InteractiveSession()
# Input tensor from Residual Network Model
with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32, shape=[None, 8, 8, 64])
with tf.name_scope('Label'):
    y_ = tf.placeholder(tf.float32, shape=[None, 500])
with tf.name_scope('label_idx'):
    y_label_idx = tf.placeholder(tf.int32, shape=[None])


# Defining weight and bias variables
def weight_variable(shape):
    initial = tf.random_uniform(shape, -1, 1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

##########################
### Build DEViSE model ###
##########################
with tf.name_scope('M1'):
    

    x_image_flat = tf.reshape(x, [-1, 4096])

    keep_prob = tf.placeholder(tf.float32)##
    y_conv_drop = tf.nn.dropout(x_image_flat, keep_prob=keep_prob)##
    

    W_fc1 = weight_variable([4096, 500]) 
    b_fc1 = bias_variable([500])
    y_conv = tf.matmul(y_conv_drop, W_fc1) + b_fc1

    output = tf.nn.l2_normalize(y_conv, 1)##

    '''
    keep_prob = tf.placeholder(tf.float32)
    y_conv_drop = tf.nn.dropout(y_conv, keep_prob=keep_prob)
    output = tf.nn.l2_normalize(y_conv, 1)
	'''

### Loss function given by DEViSE paper
tLabelMV = tf.diag_part(tf.tensordot(y_, output, axes=[[1], [1]]))
tMV = tf.matmul(fine_label, tf.transpose(output))
margin = 0.1
# loss for training
loss = tf.reduce_mean(tf.reduce_sum(tf.nn.relu(margin-tLabelMV+tMV),0),0)
training_loss = tf.summary.scalar("Training Loss", loss)
# loss for validation set
loss_valid = tf.reduce_mean(tf.reduce_sum(tf.nn.relu(margin-tLabelMV+tMV),0),0)
validation_loss = tf.summary.scalar("Validation Loss", loss_valid)


### Similarity and Accuracy
# find the closest (cosine distance) vector representations
predict_label = tf.argmax(tMV, 0)
predict_label = tf.cast(predict_label, tf.int32)
predict_label = tf.reshape(predict_label,[-1,1])
y_label_idx_temp = tf.reshape(y_label_idx, [-1,1])
# calculation of accuracy
acc,acc_op = tf.metrics.accuracy(labels = y_label_idx, predictions = predict_label, weights=None, metrics_collections=None, updates_collections=None, name=None)
tf.summary.scalar("Train Accuracy", acc)


### Regression
train_step1 =tf.train.AdamOptimizer(0.3).minimize(loss)
train_step2 =tf.train.AdamOptimizer(0.2).minimize(loss)
train_step3 =tf.train.AdamOptimizer(0.1).minimize(loss)


### Saving checkpoints
saver = tf.train.Saver()
save_dir = 'checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_validation')


### Session initialization
sess.run(tf.global_variables_initializer())

sess.run(tf.local_variables_initializer())


# Output visualized graph
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(logs_path, graph = tf.get_default_graph())


### Definition a function to create a mini-batch
# Return a total of `num` random samples and labels. 
'''

def next_batch(num, data, labels, labels_idx):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    labels_idx_shuffle = [labels_idx[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle), np.asarray(labels_idx_shuffle)

loss_list = []  # Store the values of training loss and validation loss
new_epoch = False
epoch = 0
batch_count = 0
#for i in range(1000000):
while epoch <= epoch_num:
    # Print "accuracy", "training loss" and "validation loss" at certain iterations
    if new_epoch:
        print("Epoch %d"%(epoch))

        train_accuracy = sess.run(acc_op, feed_dict = {x: x_batch, y_: y_batch_vec, y_label_idx: y_batch_idx, keep_prob: 1.0})
        print("training accuracy %g"%(train_accuracy))
        train_loss = sess.run(loss, feed_dict = {x: x_batch, y_: y_batch_vec, y_label_idx: y_batch_idx, keep_prob: 1.0})
        print("training loss: %g"%(train_loss))

        validation_accuracy = sess.run(acc_op, feed_dict = {x: X_impred_valid, y_: Y_valid_vec, y_label_idx: Y_valid, keep_prob: 1.0})
        print("validation accuracy %g"%(validation_accuracy))    # Test the model
        validation_loss = sess.run(loss_valid, feed_dict = {x: X_impred_valid, y_: Y_valid_vec, y_label_idx: Y_valid, keep_prob: 1.0})
        print("validation loss: %g"%(validation_loss))
        print("======================================")
        #print("validation accuracy %g"%sess.run(acc_op, feed_dict = {x: X_impred_valid, y_: Y_valid_vec, y_label_idx: Y_valid, keep_prob: 1.0}))    # Test the model

        loss_list.append([epoch, train_loss, validation_loss])
        summary = sess.run(merged, feed_dict = {x: x_batch, y_: y_batch_vec, y_label_idx: y_batch_idx, keep_prob: 1.0})
        writer.add_summary(summary, epoch)
        writer.flush()

        # Save checkpoints at certain iterations
        saver.save(sess=sess, save_path=save_path)

        ### 10-fold validation:
        nth_fold = epoch%10
        i = np.arange(0, len(X_impred), dtype=int)
        fold_min = int(len(X_impred)*nth_fold*0.1)
        fold_max = int(len(X_impred)*(nth_fold+1)*0.1)
        X_impred_train = np.concatenate((X_impred[np.where(i<fold_min)], X_impred[np.where(i>=fold_max)]))
        Y_train_vec = np.concatenate((Y_vec[np.where(i<fold_min)], Y_vec[np.where(i>=fold_max)]))
        Y_train = np.concatenate((Y[np.where(i<fold_min)], Y[np.where(i>=fold_max)]))

        X_impred_valid = X_impred[fold_min:fold_max]
        Y_valid_vec = Y_vec[fold_min:fold_max]
        Y_valid = Y[fold_min:fold_max]

        idx = np.arange(0 , len(X_impred_train)) 	# shullfed index for mini-batch
        np.random.shuffle(idx)						# shullfed index for mini-batch
        epoch = epoch + 1
        batch_count = 0
        new_epoch = False
    # Load the next batch of trainning set
    # x_batch, y_batch_vec, y_batch_idx = next_batch(batch_size, X_impred_train, Y_train_vec, Y_train)
    elif epoch == 0 and new_epoch==False:
        ### 10-fold validation:
        nth_fold = epoch%10
        i = np.arange(0, len(X_impred), dtype=int)
        fold_min = int(len(X_impred)*nth_fold*0.1)
        fold_max = int(len(X_impred)*(nth_fold+1)*0.1)
        X_impred_train = np.concatenate((X_impred[np.where(i<fold_min)], X_impred[np.where(i>=fold_max)]))
        Y_train_vec = np.concatenate((Y_vec[np.where(i<fold_min)], Y_vec[np.where(i>=fold_max)]))
        Y_train = np.concatenate((Y[np.where(i<fold_min)], Y[np.where(i>=fold_max)]))

        X_impred_valid = X_impred[fold_min:fold_max]
        Y_valid_vec = Y_vec[fold_min:fold_max]
        Y_valid = Y[fold_min:fold_max]

        idx = np.arange(0 , len(X_impred_train)) 	# shullfed index for mini-batch
        np.random.shuffle(idx)						# shullfed index for mini-batch
        new_epoch = False
        epoch = 1

    if batch_size*(batch_count+1) < len(X_impred_train):
    	idx_this_batch = idx[batch_size*batch_count: batch_size*(batch_count+1)]
    	batch_count = batch_count+1
    else:
    	idx_this_batch = idx[batch_size*batch_count:]
    	new_epoch = True
    	
    x_batch = [X_impred_train[i] for i in idx_this_batch]
    y_batch_vec = [Y_train_vec[i] for i in idx_this_batch]
    y_batch_idx = [Y_train[i] for i in idx_this_batch]

    x_batch = np.asarray(x_batch)
    y_batch_vec = np.asarray(y_batch_vec)
    y_batch_idx = np.asarray(y_batch_idx)

    # Train the model with larger learning rate
    if epoch < epoch_num*0.4:
        train_step1.run(feed_dict={x: x_batch, y_: y_batch_vec, y_label_idx: y_batch_idx, keep_prob: 0.999})

    # Train the model with intermediate learning rate    
    elif epoch < epoch_num*0.8 and epoch >= epoch_num*0.4:
        train_step2.run(feed_dict={x: x_batch, y_: y_batch_vec, y_label_idx: y_batch_idx, keep_prob: 0.999})

    # Train the model with smaller learning rate    
    elif epoch < epoch_num and epoch >= epoch_num*0.8:
        train_step3.run(feed_dict={x: x_batch, y_: y_batch_vec, y_label_idx: y_batch_idx, keep_prob: 0.999})    

# Save the "loss_list"
with open("loss_list.csv", "w") as f:
	writer = csv.writer(f)
	writer.writerows(loss_list)

# Estimate the accuracy for test set
print("test accuracy %g"%sess.run(acc_op, feed_dict = {x: testX_impred, y_: testY_vec, y_label_idx: testY, keep_prob: 1.0}))    # Test the model


# Close session
sess.close()