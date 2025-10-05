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
        
        return pd.DataFrame(data=data_all, columns=names_all), boston.DESCR
    
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