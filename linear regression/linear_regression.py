import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
from sklearn.preprocessing import MinMaxScaler


#data = pd.read_csv("death.csv")
data = pd.read_csv("data_examples/housing.csv")
del data["ocean_proximity"]

data = pd.DataFrame(MinMaxScaler().fit_transform(data.values), columns=data.columns, index=data.index)

print(data.columns)

#data.plot.scatter("total_rooms","median_house_value")
#plot.show()


x = data[["total_rooms","households"]]
y = data[["median_house_value"]]


#plot.scatter(x,y)
#plot.show()


def loss_function(x, y, slopes, intercept):

    """ Calculate loss function (error = (y_hat - y)^2 / n). 
    Args:
        x (pandas data frame of float): data variables
        y (pandas data frame of float): class label
        slopes (numpy list of floats): slope of the line
        intercept (float): intercept of line
    Return:
        total_error (float): loss error
    """
    total_error = 0
    for i in range(len(y)):
        if len(slopes) > 1:
             mult = slopes.dot(x.iloc[i].values)    
        else:
            mult = x.iloc[i].values[0]* slopes[0]
        total_error += ((mult+intercept) - y.iloc[i].values[0])**2
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
                gradient_m[j] = gradient_m[j] + (-(2/n) * x.iloc[i].values[j] * (y.iloc[i].values[0] - (mult + intercept)))
        else:
            # 1D data
            mult = x.iloc[i].values[0]* slopes[0]
            gradient_m[0] = gradient_m[0] + (-(2/n) * x.iloc[i].values[0] * (y.iloc[i].values[0] - (mult + intercept)))
            
        gradient_b += (-(2/n) * (y.iloc[i].values[0] - (mult + intercept)))
    
    m = np.zeros(num_features) 
    for i in range(num_features):
        m[i] = slopes[i] - (gradient_m[i] * L)

    b = intercept - gradient_b * L
    return m, b


epochs = 100
L = 0.9
m= np.zeros(len(x.columns))
b = 0
for i in range(epochs):
    if i % 20 == 0:
        print("epochs = ", i) 
    
    print(loss_function(x,y,m,b))
    m ,b = gradient_descent(x,y,m,b,L)
    print(m,b)


print("....................................")
print(m,b)
plot.scatter(x,y)
plot.plot(list(np.arange(0.0, 1.0, 0.1)), [m * x + b for x in list(np.arange(0.0, 1.0, 0.1))], color = "red")
plot.show()