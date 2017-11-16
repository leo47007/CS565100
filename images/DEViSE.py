from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow as tf
import os
import tflearn

X_impred_train = np.load('X_impred_train.npy')
X_impred_valid = np.load('X_impred_valid.npy')
testX_impred = np.load('testX_impred.npy')
Y_train_vec = np.load('Y_train_vec.npy')
Y_valid_vec = np.load('Y_valid_vec.npy')
testY_vec = np.load('testY_vec.npy')

### Environment settings###
# Setting Parameters
logs_path = 'TensorBoard/'
n_features = 4096 # number of image pixels 
batch_size = 100 # Size of a mini-batch

# Launching InteractiveSession
sess = tf.InteractiveSession()
with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32, shape=[None, 8, 8, 64])
with tf.name_scope('Label'):
    y_ = tf.placeholder(tf.int32, shape=[None, 500])

# Defining weight and bias variables
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


### Build DEViSE model
with tf.name_scope('M'):
    W_fc1 = weight_variable([4096, 500]) 
    b_fc1 = bias_variable([500])
    x_image_flat = tf.reshape(x, [-1, 4096])
    y_conv = tf.nn.relu(tf.matmul(x_image_flat, W_fc1) + b_fc1)

# Regression
with tf.name_scope('hinge_loss'):
    hinge_loss = tf.losses.hinge_loss(labels=y_, logits=y_conv)
    tf.summary.scalar("hinge_loss", hinge_loss)
train_step = tf.train.AdamOptimizer(1e-4).minimize(hinge_loss)

#Calculating the average loss of the model
with tf.name_scope('avg_loss'):
    avg_loss = tf.reduce_mean(tf.cast(hinge_loss, tf.float32))
    tf.summary.scalar("avg_loss", avg_loss)


# Saving checkpoints
saver = tf.train.Saver()
save_dir = 'checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_validation')

# Session initialization
sess.run(tf.global_variables_initializer())
tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=False)

# Output visualized graph
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(logs_path, graph = tf.get_default_graph())

# Defining a function to create a mini-batch
def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

# Initializing parameters for early stop
global total_iterations    # Total iterations
global best_validation_accuracy    # Best validation accuracy
global last_improvement    # last iteration with improvement
best_validation_accuracy = 0.0    # Recent best validation accuracy
last_improvement = 0     # last iteration with improvement
require_improvement = 1000    # If no improvements have done within 1000 iterations, stop trainning.

for i in range(20000):
    x_batch, y_batch = next_batch(batch_size, X_impred_train, Y_train_vec)    # Loading in the next batch of trainning set
    total_iterations = i
    if i%(2 * batch_size) == 0:
        train_loss = avg_loss.eval(feed_dict = {x: x_batch, y_: y_batch})
        print("step %d, training loss %g"%(i, train_loss))
        summary = sess.run(merged, feed_dict = {x: x_batch, y_: y_batch})
        writer.add_summary(summary, i)
        writer.flush()
    train_step.run(feed_dict={x: x_batch, y_: y_batch})    # Train the model

print("test loss %g"%avg_loss.eval(feed_dict = {x: testX_impred, y_: testY_vec}))    # Test the model

# Close session
sess.close()