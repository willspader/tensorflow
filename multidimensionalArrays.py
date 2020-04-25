import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

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

def multiDimensionalArraysOperations():
    graph3 = tf.Graph()
    with graph3.as_default():
        Matrix_one = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
        Matrix_two = tf.constant([[2,2,2],[2,2,2],[2,2,2]])

        add_1_operation = tf.add(Matrix_one, Matrix_two)
        add_2_operation = Matrix_one + Matrix_two

    with tf.compat.v1.Session(target='', graph=graph3, config=None) as sess:
        result = sess.run(add_1_operation)
        print ("Defined using tensorflow function :")
        print(result)
        result = sess.run(add_2_operation)
        print ("Defined using normal expressions :")
        print(result)

def matMul():
    graph4 = tf.Graph()
    with graph4.as_default():
        Matrix_one = tf.constant([[2,3],[3,4]])
        Matrix_two = tf.constant([[2,3],[3,4]])

        mul_operation = tf.matmul(Matrix_one, Matrix_two)

    with tf.compat.v1.Session(target='', graph=graph4, config=None) as sess:
        result = sess.run(mul_operation)
        print ("Defined using tensorflow function :")
        print(result)

if __name__ == '__main__':
    print('================ BEGIN MULTIDIMENSIONAL ARRAYS EXECUTION ================')
    multidimensionalArrays()
    print('================ END MULTIDIMENSIONAL ARRAYS EXECUTION ================')
    print()
    print('================ BEGIN MULTIDIMENSIONAL ARRAYS OPERATIONS EXECUTION ================')
    multiDimensionalArraysOperations()
    print('================ END MULTIDIMENSIONAL ARRAYS OPERATIONS EXECUTION ================')
    print('================ BEGIN MAT MUL EXECUTION ================')
    matMul()
    print('================ END MAT MUL EXECUTION ================')

