import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def helloWorld():
    graph1 = tf.Graph() # cria o grafo
    
    with graph1.as_default():
        a = tf.constant([2], name = 'constant_a') # definindo constante
        b = tf.constant([3], name = 'constant_b')
        print(f'a variable = {a}')
        print(f'b variable = {b}')
    sess = tf.compat.v1.Session(target='', graph=graph1, config=None)
    result = sess.run(a)
    print(result)
    sess.close()

    with graph1.as_default():
        c = tf.add(a, b) # c = a + b
        print(f'c variable = {c}')
    sess = tf.compat.v1.Session(target='', graph=graph1, config=None)
    result = sess.run(c)
    print(result)
    sess.close()

def multidimensionalArrays():
    graph2 = tf.Graph()
    with graph2.as_default():
        Scalar = tf.constant(2)
        Vector = tf.constant([5,6,2])
        Matrix = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
        Tensor = tf.constant( [ [[1,2,3],[2,3,4],[3,4,5]] , [[4,5,6],[5,6,7],[6,7,8]] , [[7,8,9],[8,9,10],[9,10,11]] ] )
    with tf.compat.v1.Session(target='', graph=graph2, config=None) as sess:
        result = sess.run(Scalar)
        print ("Scalar (1 entry):\n %s \n" % result)
        result = sess.run(Vector)
        print ("Vector (3 entries) :\n %s \n" % result)
        result = sess.run(Matrix)
        print ("Matrix (3x3 entries):\n %s \n" % result)
        result = sess.run(Tensor)
        print ("Tensor (3x3x3 entries) :\n %s \n" % result)
        
if __name__ == '__main__':
    print('================ BEGIN HELLO WORLD EXECUTION ================')
    helloWorld()
    print('================ END HELLO WORLD EXECUTION ================')
    print()
    print('================ BEGIN MULTIDIMENSIONAL ARRAYS EXECUTION ================')
    multidimensionalArrays()
    print('================ END MULTIDIMENSIONAL ARRAYS EXECUTION ================')
