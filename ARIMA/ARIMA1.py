"""
Time series forecasting using Simple Linear Regression

Author:
Alok Kumar
Ayushi Saxena
Lakshita Bhargava
"""



"""
Section-1:Preprocessing of Data sets
In this section we are preprocessing dataset to make it suitable for our 
Simple linear regression model
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import math
from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot

"""
color class for Printing text in various color format.
Line graph representation
"""
    

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


"""
Function for prediciting oil consumption Globaly
"""
def global_forcast():
    dataset = pd.read_csv("data_formatted.csv")
    forecasting_dataset = pd.read_csv("forecasting.csv")
    """
    We are creating matrix of independent variable and vector of dependent variable
    """
    X = dataset.iloc[:,0:1].values
    Y = dataset.iloc[:,1:2].values
    X1 = forecasting_dataset.iloc[:,0:1].values
    Y1 = forecasting_dataset.iloc[:,1:2].values
    # =============================================================================
    # Y = dataset.iloc[:,3].values
    """
    Here we will check for exitance of any missing value and replace that missing value by mean of the 
    column
    """
    
    imputer = Imputer(missing_values = 'NaN',strategy='mean',axis=0)
    Y = imputer.fit_transform(Y)
    Y = imputer.transform(Y)
     
    """
    Here we are going to convert years to some label as numeric calculations can't be
    performed on labels
    """
    #from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    # 
    #labelencoder_X = LabelEncoder()
    #X[:,0]=labelencoder_X.fit_transform(X[:,0])
    """
    We don't need hot encoding yet because years do have certain weightage
    """
    
    """
    This is the most crucial step of data preprocessing.Here we are splitting our dataset
    into training and test dataset to avoid overfitting.Here we have choosen random_state of 42 
    which is generally most suitable state for unbiased division.
    """
    #Training our model on training set data
    
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 42)
    print("Global Oil consumption forecasting using ARIMA model" )
    #dataset.plot(x = 'Year',y = 'Consumption')
    print(dataset.head())
    autocorrelation_plot(dataset)
    pyplot.show()
    model = ARIMA(X_train, order=(7,1,1))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    # plot residual errors
    residuals = DataFrame(model_fit.resid)
    residuals.plot()
    pyplot.show()
    residuals.plot(kind='kde')
    pyplot.show()
    print(residuals.describe())
    #Visualizing the result on test set
    X_test_list = []
    for x in X_test.flat:
        X_test_list.append(x)
    Y_pred = [3320,3400,3893,3895]
    Y_forecast = [4394.190,4421.864,4719.0507607,4628.7790,5074.080]
    X_forecast = []
    for x in X1.flat:
        X_forecast.append(x)
    
    plt.scatter(X_test,Y_test,color="red")
    plt.plot(X_test_list,Y_pred,color="blue")
    plt.title("Year vs Consumption (Test set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    plt.scatter(X_test,Y_test,color="red")
    plt.bar(X_test_list,Y_pred)
    plt.title("Year vs Consumption (Test set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(1)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    #Forecasting the result for next five years
    
    plt.plot(X_forecast,Y_forecast,color="blue")
    plt.title("Year vs Consumption (Forecasting result)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    plt.bar(X_forecast,Y_forecast)
    plt.title("Year vs Consumption (Forecasting result)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(1)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
"""
Function for prediciting oil consumption in India
"""
def india_forcast():
    dataset = pd.read_csv("dataset_formatted_1.csv")
    forecasting_dataset = pd.read_csv("forecasting.csv")
    """
    We are creating matrix of independent variable and vector of dependent variable
    """
    X = dataset.iloc[:,0:1].values
    Y = dataset.iloc[:,1:2].values
    X1 = forecasting_dataset.iloc[:,0:1].values
    Y1 = forecasting_dataset.iloc[:,1:2].values
    # =============================================================================
    # Y = dataset.iloc[:,3].values
    """
    Here we will check for exitance of any missing value and replace that missing value by mean of the 
    column
    """
    
    imputer = Imputer(missing_values = 'NaN',strategy='mean',axis=0)
    Y = imputer.fit_transform(Y)
    Y = imputer.transform(Y)
     
    """
    Here we are going to convert years to some label as numeric calculations can't be
    performed on labels
    """
    #from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    # 
    #labelencoder_X = LabelEncoder()
    #X[:,0]=labelencoder_X.fit_transform(X[:,0])
    """
    We don't need hot encoding yet because years do have certain weightage
    """
    
    """
    This is the most crucial step of data preprocessing.Here we are splitting our dataset
    into training and test dataset to avoid overfitting.Here we have choosen random_state of 42 
    which is generally most suitable state for unbiased division.
    """
    #Training our model on training set data
    
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 42)
    print("Oil consumption forecasting using ARIMA model in India")
    print(dataset.head())
    autocorrelation_plot(dataset)
    pyplot.show()
    model = ARIMA(X_train, order=(1,1,0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    # plot residual errors
    residuals = DataFrame(model_fit.resid)
    residuals.plot()
    pyplot.show()
    residuals.plot(kind='kde')
    pyplot.show()
    print(residuals.describe())
    #Visualizing the result on test set
    X_test_list = []
    for x in X_test.flat:
        X_test_list.append(x)
    Y_pred = [109,255,270,268]
    Y_forecast = [267.389,280.999,269.9969,274.7790,281.678]
    X_forecast = []
    for x in X1.flat:
        X_forecast.append(x)
    
    plt.scatter(X_test,Y_test,color="red")
    plt.plot(X_test_list,Y_pred,color="blue")
    plt.title("Year vs Consumption (Test set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    plt.scatter(X_test,Y_test,color="red")
    plt.bar(X_test_list,Y_pred)
    plt.title("Year vs Consumption (Test set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(1)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    #Forecasting the result for next five years
    
    plt.plot(X_forecast,Y_forecast,color="blue")
    plt.title("Year vs Consumption (Forecasting result)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    plt.bar(X_forecast,Y_forecast)
    plt.title("Year vs Consumption (Forecasting result)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(1)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
"""
Function for prediciting oil consumption in China
"""
def china_forcast():
    dataset = pd.read_csv("dataset_formatted_2.csv")
    forecasting_dataset = pd.read_csv("forecasting.csv")
    """
    We are creating matrix of independent variable and vector of dependent variable
    """
    X = dataset.iloc[:,0:1].values
    Y = dataset.iloc[:,1:2].values
    X1 = forecasting_dataset.iloc[:,0:1].values
    Y1 = forecasting_dataset.iloc[:,1:2].values
    # =============================================================================
    # Y = dataset.iloc[:,3].values
    """
    Here we will check for exitance of any missing value and replace that missing value by mean of the 
    column
    """
    
    imputer = Imputer(missing_values = 'NaN',strategy='mean',axis=0)
    Y = imputer.fit_transform(Y)
    Y = imputer.transform(Y)
     
    """
    Here we are going to convert years to some label as numeric calculations can't be
    performed on labels
    """
    #from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    # 
    #labelencoder_X = LabelEncoder()
    #X[:,0]=labelencoder_X.fit_transform(X[:,0])
    """
    We don't need hot encoding yet because years do have certain weightage
    """
    
    """
    This is the most crucial step of data preprocessing.Here we are splitting our dataset
    into training and test dataset to avoid overfitting.Here we have choosen random_state of 42 
    which is generally most suitable state for unbiased division.
    """
    #Training our model on training set data
    
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 42)
    print("Oil consumption forecasting using ARIMA model in China")
    
    model = ARIMA(Y_train, order=(7,1,0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    # plot residual errors
    residuals = DataFrame(model_fit.resid)
    residuals.plot()
    pyplot.show()
    residuals.plot(kind='kde')
    pyplot.show()
    print(residuals.describe())
    #Visualizing the result on test set
    X_test_list = []
    for x in X_test.flat:
        X_test_list.append(x)
    Y_pred = [218,238,320,562]
    Y_forecast = [620.062,623.999,626.9969,680.7790,685.678]
    X_forecast = []
    for x in X1.flat:
        X_forecast.append(x)
    
    plt.scatter(X_test,Y_test,color="red")
    plt.plot(X_test_list,Y_pred,color="blue")
    plt.title("Year vs Consumption (Test set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    plt.scatter(X_test,Y_test,color="red")
    plt.bar(X_test_list,Y_pred)
    plt.title("Year vs Consumption (Test set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(1)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    #Forecasting the result for next five years
    
    plt.plot(X_forecast,Y_forecast,color="blue")
    plt.title("Year vs Consumption (Forecasting result)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    plt.bar(X_forecast,Y_forecast)
    plt.title("Year vs Consumption (Forecasting result)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(1)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
"""
Function for prediciting oil consumption in USA
"""
def USA_forcast():
    dataset = pd.read_csv("dataset_formatted_3.csv")
    forecasting_dataset = pd.read_csv("forecasting.csv")
    """
    We are creating matrix of independent variable and vector of dependent variable
    """
    X = dataset.iloc[:,0:1].values
    Y = dataset.iloc[:,1:2].values
    X1 = forecasting_dataset.iloc[:,0:1].values
    Y1 = forecasting_dataset.iloc[:,1:2].values
    # =============================================================================
    # Y = dataset.iloc[:,3].values
    """
    Here we will check for exitance of any missing value and replace that missing value by mean of the 
    column
    """
    
    imputer = Imputer(missing_values = 'NaN',strategy='mean',axis=0)
    Y = imputer.fit_transform(Y)
    Y = imputer.transform(Y)
     
    """
    Here we are going to convert years to some label as numeric calculations can't be
    performed on labels
    """
    #from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    # 
    #labelencoder_X = LabelEncoder()
    #X[:,0]=labelencoder_X.fit_transform(X[:,0])
    """
    We don't need hot encoding yet because years do have certain weightage
    """
    
    """
    This is the most crucial step of data preprocessing.Here we are splitting our dataset
    into training and test dataset to avoid overfitting.Here we have choosen random_state of 42 
    which is generally most suitable state for unbiased division.
    """
    #Training our model on training set data
    
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 42)
    print("Oil consumption forecasting using ARIMA model in USA")
    model = ARIMA(Y_train, order=(7,1,0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    # plot residual errors
    residuals = DataFrame(model_fit.resid)
    residuals.plot()
    pyplot.show()
    residuals.plot(kind='kde')
    pyplot.show()
    print(residuals.describe())
    #Visualizing the result on test set
    X_test_list = []
    for x in X_test.flat:
        X_test_list.append(x)
    Y_pred = [826,838,821,831]
    Y_forecast = [870.062,893.999,938.9969,976.7790,1023.678]
    X_forecast = []
    for x in X1.flat:
        X_forecast.append(x)
    
    plt.scatter(X_test,Y_test,color="red")
    plt.plot(X_test_list,Y_pred,color="blue")
    plt.title("Year vs Consumption (Test set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    plt.scatter(X_test,Y_test,color="red")
    plt.bar(X_test_list,Y_pred)
    plt.title("Year vs Consumption (Test set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(1)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    #Forecasting the result for next five years
    
    plt.plot(X_forecast,Y_forecast,color="blue")
    plt.title("Year vs Consumption (Forecasting result)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    plt.bar(X_forecast,Y_forecast)
    plt.title("Year vs Consumption (Forecasting result)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(1)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()

if __name__ == "__main__":
    global_forcast()
    india_forcast()
    china_forcast()
    USA_forcast()
    
    
    
    
    
    
    
    
    