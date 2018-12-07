#Statistics Library for statistical analysis 
#@Author Bryan Baek

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

#Histogram for Steering Data Distribution 
def generate_histogram(x,path='DataDistribution.png'): 
    fig = plt.figure()
    plt.hist(x,200, normed=1, facecolor='green', alpha=0.75)
    plt.xlabel('Wheel steering')
    plt.ylabel('Probability')
    plt.title('Histogram of training data distribution.')
    fig.savefig(path)






