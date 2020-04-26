import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

tf.disable_v2_behavior()


def linearRegressionCsv():

    df = pd.read_csv("FuelConsumptionCo2.csv")

    df.head()

    # Digamos que a gente queira prever a emissão de CO2 baseado no tamanho dos carros.
    # Para isso, vamos definir X e Y para a regressão linear, ou seja, train_x e train_y

    train_x = np.asanyarray(df[['ENGINESIZE']])
    train_y = np.asanyarray(df[['CO2EMISSIONS']])

    # Primeiro definimos as variáveis a e b com um random qualquer

    a = tf.Variable(20.0)
    b = tf.Variable(30.2)
    y = a * train_x + b

    # Agora, vamos definir uma função de perda para nossa regressão, para que possamos treinar nosso modelo para melhor ajustar nossos dados.
    # Em uma regressão linear, minimizamos o erro ao quadrado da diferença entre os valores previstos (obtidos da equação)
    # e os valores alvo (os dados que temos). Em outras palavras, queremos minimizar o quadrado dos valores previstos menos o valor alvo.
    # Então, definimos a equação a ser minimizada como perda.

    loss = tf.reduce_mean(tf.square(y - train_y))

    # taxa de aprendizado

    optimizer = tf.train.GradientDescentOptimizer(0.05)

    # método de treinamento

    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    loss_values = []
    train_data = []
    for step in range(100):
        _, loss_val, a_val, b_val = sess.run([train, loss, a, b])
        loss_values.append(loss_val)
        if step % 5 == 0:
            print(step, loss_val, a_val, b_val)
            train_data.append([a_val, b_val])
            
    plt.plot(loss_values, 'ro')

    cr, cg, cb = (1.0, 1.0, 0.0)
    for f in train_data:
        cb += 1.0 / len(train_data)
        cg -= 1.0 / len(train_data)
        if cb > 1.0: cb = 1.0
        if cg < 0.0: cg = 0.0
        [a, b] = f
        f_y = np.vectorize(lambda x: a*x + b)(train_x)
        line = plt.plot(train_x, f_y)
        plt.setp(line, color=(cr,cg,cb))

    plt.plot(train_x, train_y, 'ro')


    green_line = mpatches.Patch(color='red', label='Data Points')

    plt.legend(handles=[green_line])

    plt.show()

        
if __name__ == '__main__':
    print('================ BEGIN LINEAR REGRESSION EXECUTION ================')
    linearRegressionCsv()
    print('================ END LINEAR REGRESSION EXECUTION ================')
