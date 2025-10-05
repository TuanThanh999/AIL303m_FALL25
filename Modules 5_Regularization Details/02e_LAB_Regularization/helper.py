import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(72018)

def to_2d(array):
    return array.reshape(array.shape[0], -1)

def boston_dataframe(description=False):
    # Load Boston housing data from original source since sklearn deprecated it
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    
    names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
             'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    
    target = to_2d(target)
    
    data_all = np.concatenate([data, target], axis=1)
    names_all = np.concatenate([names, np.array(['MEDV'])], axis=0)
    
    if description:
        description_text = """
        Boston Housing Dataset
        
        Data Set Characteristics:
        - Number of Instances: 506
        - Number of Attributes: 13 numeric/categorical predictive
        - Median Value (attribute 14) is usually the target
        
        Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town.
        - CHAS     Charles River dummy variable (1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's
        """
        return pd.DataFrame(data=data_all, columns=names_all), description_text
    
    else: 
        return pd.DataFrame(data=data_all, columns=names_all)
    
def plot_exponential_data():
    data = np.exp(np.random.normal(size=1000))
    plt.hist(data)
    plt.show()
    return data
    
def plot_square_normal_data():
    data = np.square(np.random.normal(loc=5, size=1000))
    plt.hist(data)
    plt.show()
    return data