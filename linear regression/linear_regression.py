import pandas as pd
import matplotlib.pyplot as plot
import numpy as np

def loss_function(x, y, slopes, intercept):

    """ Calculate loss function (error = (y_hat - y)^2 / n). 
    
    Args:
        x (pandas data frame of float): data variables
        y (pandas data frame of float): class label
        slopes (numpy list of floats): slope of the line
        intercept (float): intercept of line
    Return:
        m (numpy list of floats): slope of the line  (theta)
        b (float): intercept
    """

    total_error = 0
    for i in range(len(y)):
        if len(slopes) > 1:
             mult = slopes.dot(x.iloc[i].values)    
        else:
            mult = x.iloc[i]* slopes[0]

        total_error += ((mult+intercept) - y.iloc[1])**2

        
        
    return total_error/len(y)

def gradient_descent(x,y,slopes, intercept, L):
    """ Calculate slopes and intercepts using gradient descent. 
    
    Args:
        x (pandas data frame of float): data variables
        y (pandas data frame of float): class label
        slopes (numpy list of floats): slope of the line
        intercept (float): intercept of line
        L (float): jump size 
    Return:
        m (numpy list of floats): slope of the line  (theta)
        b (float): intercept
    """
    gradient_b = 0
    num_features = len(slopes)
    gradient_m = np.zeros(num_features) 
    n = len(y)
    for i in range(n):
        
        if( num_features > 1):
            # More than 1D data
            mult = slopes.dot(x.iloc[i].values)
            for j in range(num_features):   
                gradient_m[j] = gradient_m[j] + (-(2/n) * x.iloc[i].values[j] * (y.iloc[i] - (mult + intercept)))
        else:
            # 1D data
            mult = x.iloc[i]* slopes[0]
            gradient_m[0] = gradient_m[0] + (-(2/n) * x.iloc[i] * (y.iloc[i] - (mult + intercept)))
            
        gradient_b += (-(2/n) * (y.iloc[i] - (mult + intercept)))
    
    m = np.zeros(num_features) 
    for i in range(num_features):
        m[i] = slopes[i] - (gradient_m[i] * L)

    b = intercept - gradient_b * L
    return np.nan_to_num(m), np.nan_to_num(b)
