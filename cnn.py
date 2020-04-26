import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow.compat.v1 as tf

import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

tf.disable_eager_execution()

def cnn():
    sess = tf.InteractiveSession()
    
    x  = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # Weight tensor
    W = tf.Variable(tf.zeros([784, 10],tf.float32))
    # Bias tensor
    b = tf.Variable(tf.zeros([10],tf.float32))

    sess.run(tf.global_variables_initializer())

    # operação matemática para somar peso e bias às entradas  
    tf.matmul(x,W) + b

    y = tf.nn.softmax(tf.matmul(x,W) + b)

    # cost function
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    #Load 50 training examples for each training iteration   
    for i in range(1000):
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    # test
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}) * 100
    print("The final accuracy for the simple ANN model is: {} % ".format(acc) )

    sess.close()
        
if __name__ == '__main__':
    print('================ BEGIN CNN EXECUTION ================')
    cnn()
    print('================ END CNN EXECUTION ================')
