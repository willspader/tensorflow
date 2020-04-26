import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 6)

def linearRegression():
    X = np.arange(0.0, 5.0, 0.1)
    a = 1
    b = 0

    Y = a * X + b 

    plt.plot(X, Y) 
    plt.ylabel('Dependent Variable')
    plt.xlabel('Independent Variable')
    plt.show()

        
if __name__ == '__main__':
    print('================ BEGIN LINEAR REGRESSION EXECUTION ================')
    linearRegression()
    print('================ END LINEAR REGRESSION EXECUTION ================')
