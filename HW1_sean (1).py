import tensorflow as tf
import numpy as np 
from sklearn.datasets import fetch_california_housing
from IPython.display import clear_output, Image, display, HTML

###### Do not modify here ###### 
def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = graph_def
    #strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))
###### Do not modify  here ######

###### Implement Data Preprocess here ######
from sklearn import preprocessing

housing = fetch_california_housing()
print("Shape of dataset:", housing.data.shape)
print("Shape of label:", housing.target.shape)

X = np.array(housing.data)
y = np.array(housing.target)

X = preprocessing.scale(X)
numData = len(housing.target)
trainRate = 0.9
numTrain = int(trainRate * numData)
numData = int(numData)

X_train = tf.constant(X[:numTrain, :], dtype='float32')
X_test = tf.constant(X[numTrain:, :], dtype='float32')
y_train = tf.constant(y[:numTrain], dtype='float32')
y_test = tf.constant(y[numTrain:], dtype='float32')

###### Implement Data Preprocess here ######

# Creating TensorFlow Structure
Weight = tf.Variable(tf.random_uniform([8, 1]))
bias = tf.Variable(tf.zeros([1]))

y_predict = tf.matmul(X_train, Weight) + bias*tf.ones([numTrain])
loss = tf.reduce_mean(tf.square(y_predict - y_train))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(loss)

# Initialize
init = tf.global_variables_initializer()

###### Start TF session ######
with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            Weight_result = sess.run(Weight)
            bias_result = sess.run(bias)
            print(step, '\n', Weight_result, bias_result)
            
    show_graph(tf.get_default_graph().as_graph_def())

###### Start TF session ######

X_train_array = np.matrix(X[:numTrain, :])
y_train_array = np.matrix(y[:numTrain])
errRate = np.mean((y_train_array - (X_train_array * Weight_result + bias_result) / y_train_array))
print('Error Rate:')
print(errRate)