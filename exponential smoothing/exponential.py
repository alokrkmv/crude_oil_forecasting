"""
Time series forecasting using Exponential Smoothing

Author:
Alok Kumar
Ayushi Saxena
Lakshita Bhargava
"""



"""
Section-1:Preprocessing of Data sets
In this section we are preprocessing dataset to make it suitable for our 
Support Vector Regression model
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import math


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
   
def global_forecast():
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
    
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 42)
    
    #Training our exponential smoothing model on training data.
    
    """
    In this part of the program we are just faltting the matrix of depeneding variable for training set
    """
#    print(X_train)
    Y_train_list = []
    for y in Y_train.flat:
        Y_train_list.append(y)
#    print(Y_train_list)
    X_train_list = []
    for x in X_train.flat:
        X_train_list.append(x)
#    print(X_train_list)
    
    """
    In this section of the code we will apply the exponential smoothing model on our training data
    Fomulae to be used:
    F(t)=β(A(t-1))+(1-β)F(t-1)
    where β is exponential smoothing constant
    Here we are taking the value of β = 0.2
    """
    x = 0.2
    Y_train_predicted = []
    counter = 0
    for y in Y_train_list:
        if counter == 0:
            Y_train_predicted.append(math.ceil(Y_train_list[counter]))
        else:
            res = x*(Y_train_list[counter-1])+(1-x)*Y_train_predicted[counter-1]
            Y_train_predicted.append(math.ceil(res))
        counter+=1
#    print( Y_train_predicted)
    print("Global oil consumption forecasting")
    
    #Visualizing training set results
    
    """
    Line graph representation
    """
    plt.scatter(X_train,Y_train,color="red")
    plt.plot(X_train,Y_train_predicted,color="blue")
    plt.title("Year vs Consumption (Training set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    """
    Bar graph representation
    """
    plt.scatter(X_train,Y_train,color="red")
    plt.bar(X_train_list,Y_train_predicted)
    plt.title("Year vs Consumption (Training set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
#    print(X_train)
    
    # Testing our exponential smoothing model on test set
    
    Y_test_list = []
    for y in Y_test.flat:
        Y_test_list.append(y)
#    print(Y_test_list)
    X_test_list = []
    for x in X_test.flat:
        X_test_list.append(x)
#    print(X_test_list)
    
    x = 0.2
    Y_test_predicted = []
    counter = 0
    for y in Y_test_list:
        if counter == 0:
            Y_test_predicted.append(math.ceil(Y_test_list[counter]))
        else:
            res = x*(Y_test_list[counter-1])+(1-x)*Y_test_predicted[counter-1]
            Y_test_predicted.append(math.ceil(res))
        counter+=1
#    print(Y_test_predicted)
    
    print("Actual Global oil Consumption")
    actual_list = []
    counter = 0
    for x in X_test_list:
        actual_list.append((x,Y_test_list[counter]))
        counter+=1
    print(actual_list)
    print("Predicted Global oil Consumption")
    predicted_list = []
    counter = 0
    for x in X_test_list:
        predicted_list.append((x,Y_test_predicted[counter]))
        counter+=1
    print(predicted_list)
    
       
    #Visualizing test set results
    
    """
    Line graph representation
    """
    plt.scatter(X_test,Y_test,color="red")
    plt.plot(X_test_list,Y_test_list,color="blue")
    plt.title("Year vs Consumption (Test set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    """
    Bar graph representation
    """
    plt.scatter(X_test,Y_test,color="red")
    plt.bar(X_test_list,Y_test_predicted)
    plt.title("Year vs Consumption (Test set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    # Forecasting the result using our exponential smoothing model
    
    Y_forecast_list = [4230.77,4269.08, 4307.39,4345.70,4384.01]
    X_forecast_list = []
    for x in X1.flat:
        X_forecast_list.append(x)
    print(X_forecast_list)
    
    x = 0.2
    Y_forecasted = []
    counter = 0
    for y in Y_forecast_list:
        if counter == 0:
            Y_forecasted.append(math.ceil(Y_forecast_list[counter]))
        else:
            res = x*(Y_test_list[counter-1])+(1-x)*Y_forecasted[counter-1]
            Y_forecasted.append(math.ceil(res))
        counter+=1
    print(Y_forecasted)
    
    
       
    
    """
    Line graph representation
    """
    plt.plot(X_forecast_list,Y_forecasted,color="blue")
    plt.title("Year vs Consumption (Forecasting)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    """
    Bar graph representation
    """
    plt.bar(X_forecast_list,Y_forecasted)
    plt.title("Year vs Consumption (Forecasting)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    """
    This function is used for prediction of oil cunsumption In india in next five years
    """
def india_forecast():
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
    
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 42)
    
    #Training our exponential smoothing model on training data.
    
    """
    In this part of the program we are just faltting the matrix of depeneding variable for training set
    """
#    print(X_train)
    Y_train_list = []
    for y in Y_train.flat:
        Y_train_list.append(y)
#    print(Y_train_list)
    X_train_list = []
    for x in X_train.flat:
        X_train_list.append(x)
#    print(X_train_list)
    
    """
    In this section of the code we will apply the exponential smoothing model on our training data
    Fomulae to be used:
    F(t)=β(A(t-1))+(1-β)F(t-1)
    where β is exponential smoothing constant
    Here we are taking the value of β = 0.2
    """
    x = 0.2
    Y_train_predicted = []
    counter = 0
    for y in Y_train_list:
        if counter == 0:
            Y_train_predicted.append(math.ceil(Y_train_list[counter]))
        else:
            res = x*(Y_train_list[counter-1])+(1-x)*Y_train_predicted[counter-1]
            Y_train_predicted.append(math.ceil(res))
        counter+=1
#    print( Y_train_predicted)
    print("Oil consumption forecasting in India\n")
    
    #Visualizing training set results
    
    """
    Line graph representation
    """
    plt.scatter(X_train,Y_train,color="red")
    plt.plot(X_train,Y_train_predicted,color="blue")
    plt.title("Year vs Consumption (Training set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    """
    Bar graph representation
    """
    plt.scatter(X_train,Y_train,color="red")
    plt.bar(X_train_list,Y_train_predicted)
    plt.title("Year vs Consumption (Training set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
#    print(X_train)
    
    # Testing our exponential smoothing model on test set
    
    Y_test_list = []
    for y in Y_test.flat:
        Y_test_list.append(y)
#    print(Y_test_list)
    X_test_list = []
    for x in X_test.flat:
        X_test_list.append(x)
#    print(X_test_list)
    
    x = 0.2
    Y_test_predicted = []
    counter = 0
    for y in Y_test_list:
        if counter == 0:
            Y_test_predicted.append(math.ceil(Y_test_list[counter]))
        else:
            res = x*(Y_test_list[counter-1])+(1-x)*Y_test_predicted[counter-1]
            Y_test_predicted.append(math.ceil(res))
        counter+=1
#    print(Y_test_predicted)
    
    print("Actual oil Consumption in India")
    actual_list = []
    counter = 0
    for x in X_test_list:
        actual_list.append((x,Y_test_list[counter]))
        counter+=1
    print(actual_list)
    print("Predicted  oil Consumption in India")
    predicted_list = []
    counter = 0
    for x in X_test_list:
        predicted_list.append((x,Y_test_predicted[counter]))
        counter+=1
    print(predicted_list)
    
      #Visualizing test set result 
    
    """
    Line graph representation
    """
    plt.scatter(X_test,Y_test,color="red")
    plt.plot(X_test_list,Y_test_list,color="blue")
    plt.title("Year vs Consumption (Test set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    """
    Bar graph representation
    """
    plt.scatter(X_test,Y_test,color="red")
    plt.bar(X_test_list,Y_test_predicted)
    plt.title("Year vs Consumption (Test set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    # Forecasting the result using our exponential smoothing model
    
    Y_forecast_list = [267,268,269,270,271]
    X_forecast_list = []
    for x in X1.flat:
        X_forecast_list.append(x)
    print(X_forecast_list)
    
    x = 0.2
    Y_forecasted = []
    counter = 0
    for y in Y_forecast_list:
        if counter == 0:
            Y_forecasted.append(math.ceil(Y_forecast_list[counter]))
        else:
            res = x*(Y_test_list[counter-1])+(1-x)*Y_forecasted[counter-1]
            Y_forecasted.append(math.ceil(res))
        counter+=1
    print(Y_forecasted)
    
    
       
    #Visualizing test set results
    
    """
    Line graph representation
    """
    plt.plot(X_forecast_list,Y_forecasted,color="blue")
    plt.title("Year vs Consumption (Forecasting)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    """
    Bar graph representation
    """
    plt.bar(X_forecast_list,Y_forecasted)
    plt.title("Year vs Consumption (Forecasting)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    """
    This function is used for prediction of oil cunsumption In China in next five years
    """
def china_forecast():
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
    
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 42)
    
    #Training our exponential smoothing model on training data.
    
    """
    In this part of the program we are just faltting the matrix of depeneding variable for training set
    """
#    print(X_train)
    Y_train_list = []
    for y in Y_train.flat:
        Y_train_list.append(y)
#    print(Y_train_list)
    X_train_list = []
    for x in X_train.flat:
        X_train_list.append(x)
#    print(X_train_list)
    
    """
    In this section of the code we will apply the exponential smoothing model on our training data
    Fomulae to be used:
    F(t)=β(A(t-1))+(1-β)F(t-1)
    where β is exponential smoothing constant
    Here we are taking the value of β = 0.2
    """
    x = 0.2
    Y_train_predicted = []
    counter = 0
    for y in Y_train_list:
        if counter == 0:
            Y_train_predicted.append(math.ceil(Y_train_list[counter]))
        else:
            res = x*(Y_train_list[counter-1])+(1-x)*Y_train_predicted[counter-1]
            Y_train_predicted.append(math.ceil(res))
        counter+=1
#    print( Y_train_predicted)
    print("Oil consumption forecasting in China\n")
    
    #Visualizing training set results
    
    """
    Line graph representation
    """
    plt.scatter(X_train,Y_train,color="red")
    plt.plot(X_train,Y_train_predicted,color="blue")
    plt.title("Year vs Consumption (Training set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    """
    Bar graph representation
    """
    plt.scatter(X_train,Y_train,color="red")
    plt.bar(X_train_list,Y_train_predicted)
    plt.title("Year vs Consumption (Training set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
#    print(X_train)
    
    # Testing our exponential smoothing model on test set
    
    Y_test_list = []
    for y in Y_test.flat:
        Y_test_list.append(y)
#    print(Y_test_list)
    X_test_list = []
    for x in X_test.flat:
        X_test_list.append(x)
#    print(X_test_list)
    
    x = 0.2
    Y_test_predicted = []
    counter = 0
    for y in Y_test_list:
        if counter == 0:
            Y_test_predicted.append(math.ceil(Y_test_list[counter]))
        else:
            res = x*(Y_test_list[counter-1])+(1-x)*Y_test_predicted[counter-1]
            Y_test_predicted.append(math.ceil(res))
        counter+=1
#    print(Y_test_predicted)
    
    print("Actual oil Consumption in China")
    actual_list = []
    counter = 0
    for x in X_test_list:
        actual_list.append((x,Y_test_list[counter]))
        counter+=1
    print(actual_list)
    print("Predicted  oil Consumption in China")
    predicted_list = []
    counter = 0
    for x in X_test_list:
        predicted_list.append((x,Y_test_predicted[counter]))
        counter+=1
    print(predicted_list)
    
      #Visualizing test set result 
    
    """
    Line graph representation
    """
    plt.scatter(X_test,Y_test,color="red")
    plt.plot(X_test_list,Y_test_list,color="blue")
    plt.title("Year vs Consumption (Test set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    """
    Bar graph representation
    """
    plt.scatter(X_test,Y_test,color="red")
    plt.bar(X_test_list,Y_test_predicted)
    plt.title("Year vs Consumption (Test set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    # Forecasting the result using our exponential smoothing model
    
    Y_forecast_list = [604,628,653,677,702]
    X_forecast_list = []
    for x in X1.flat:
        X_forecast_list.append(x)
    print(X_forecast_list)
    
    x = 0.2
    Y_forecasted = []
    counter = 0
    for y in Y_forecast_list:
        if counter == 0:
            Y_forecasted.append(math.ceil(Y_forecast_list[counter]))
        else:
            res = x*(Y_test_list[counter-1])+(1-x)*Y_forecasted[counter-1]
            Y_forecasted.append(math.ceil(res))
        counter+=1
    print(Y_forecasted)
    
    
       
    #Visualizing test set results
    
    """
    Line graph representation
    """
    plt.plot(X_forecast_list,Y_forecasted,color="blue")
    plt.title("Year vs Consumption (Forecasting)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    """
    Bar graph representation
    """
    plt.bar(X_forecast_list,Y_forecasted)
    plt.title("Year vs Consumption (Forecasting)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    """
    This function is used for prediction of oil cunsumption In China in next five years
    """
def USA_forecast():
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
    
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 42)
    
    #Training our exponential smoothing model on training data.
    
    """
    In this part of the program we are just faltting the matrix of depeneding variable for training set
    """
#    print(X_train)
    Y_train_list = []
    for y in Y_train.flat:
        Y_train_list.append(y)
#    print(Y_train_list)
    X_train_list = []
    for x in X_train.flat:
        X_train_list.append(x)
#    print(X_train_list)
    
    """
    In this section of the code we will apply the exponential smoothing model on our training data
    Fomulae to be used:
    F(t)=β(A(t-1))+(1-β)F(t-1)
    where β is exponential smoothing constant
    Here we are taking the value of β = 0.2
    """
    x = 0.2
    Y_train_predicted = []
    counter = 0
    for y in Y_train_list:
        if counter == 0:
            Y_train_predicted.append(math.ceil(Y_train_list[counter]))
        else:
            res = x*(Y_train_list[counter-1])+(1-x)*Y_train_predicted[counter-1]
            Y_train_predicted.append(math.ceil(res))
        counter+=1
#    print( Y_train_predicted)
    print("Oil consumption forecasting in USA\n")
    
    #Visualizing training set results
    
    """
    Line graph representation
    """
    plt.scatter(X_train,Y_train,color="red")
    plt.plot(X_train,Y_train_predicted,color="blue")
    plt.title("Year vs Consumption (Training set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    """
    Bar graph representation
    """
    plt.scatter(X_train,Y_train,color="red")
    plt.bar(X_train_list,Y_train_predicted)
    plt.title("Year vs Consumption (Training set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
#    print(X_train)
    
    # Testing our exponential smoothing model on test set
    
    Y_test_list = []
    for y in Y_test.flat:
        Y_test_list.append(y)
#    print(Y_test_list)
    X_test_list = []
    for x in X_test.flat:
        X_test_list.append(x)
#    print(X_test_list)
    
    x = 0.2
    Y_test_predicted = []
    counter = 0
    for y in Y_test_list:
        if counter == 0:
            Y_test_predicted.append(math.ceil(Y_test_list[counter]))
        else:
            res = x*(Y_test_list[counter-1])+(1-x)*Y_test_predicted[counter-1]
            Y_test_predicted.append(math.ceil(res))
        counter+=1
#    print(Y_test_predicted)
    
    print("Actual oil Consumption in USA")
    actual_list = []
    counter = 0
    for x in X_test_list:
        actual_list.append((x,Y_test_list[counter]))
        counter+=1
    print(actual_list)
    print("Predicted  oil Consumption in USA")
    predicted_list = []
    counter = 0
    for x in X_test_list:
        predicted_list.append((x,Y_test_predicted[counter]))
        counter+=1
    print(predicted_list)
    
      #Visualizing test set result 
    
    """
    Line graph representation
    """
    plt.scatter(X_test,Y_test,color="red")
    plt.plot(X_test_list,Y_test_list,color="blue")
    plt.title("Year vs Consumption (Test set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    """
    Bar graph representation
    """
    plt.scatter(X_test,Y_test,color="red")
    plt.bar(X_test_list,Y_test_predicted)
    plt.title("Year vs Consumption (Test set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    # Forecasting the result using our exponential smoothing model
    
    Y_forecast_list = [812.85,812.61,812.38,812.15,811.91]
    X_forecast_list = []
    for x in X1.flat:
        X_forecast_list.append(x)
    print(X_forecast_list)
    
    x = 0.2
    Y_forecasted = []
    counter = 0
    for y in Y_forecast_list:
        if counter == 0:
            Y_forecasted.append(math.ceil(Y_forecast_list[counter]))
        else:
            res = x*(Y_test_list[counter-1])+(1-x)*Y_forecasted[counter-1]
            Y_forecasted.append(math.ceil(res))
        counter+=1
    print(Y_forecasted)
    
    
       
    #Visualizing test set results
    
    """
    Line graph representation
    """
    plt.plot(X_forecast_list,Y_forecasted,color="blue")
    plt.title("Year vs Consumption (Forecasting)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    """
    Bar graph representation
    """
    plt.bar(X_forecast_list,Y_forecasted)
    plt.title("Year vs Consumption (Forecasting)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
if __name__=="__main__":
    global_forecast()
    india_forecast()
    china_forecast()
    USA_forecast()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    