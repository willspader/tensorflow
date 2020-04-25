import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def placeholder():
    tf.compat.v1.disable_eager_execution()
    a = tf.compat.v1.placeholder(tf.compat.v1.float32)
    b = a * 2
    with tf.compat.v1.Session() as sess:
        result = sess.run(b,feed_dict={a:3.5})
        print (result)
    

if __name__ == '__main__':
    print('================ BEGIN PLACEHOLDER EXECUTION ================')
    placeholder()
    print('================ END PLACEHOLDER ================')

