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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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
    
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 42)
    
    """
    In this section we are going to fit the simple regression model to our training set
    For this purpose we are using Linear Regression module inbuild in sklearn library
    
    """
    
    regressor = LinearRegression()
    regressor.fit(X_train,Y_train)
    
    """
    After training our model on training data set we need to 
    test it on test_data set to see whether it is efficient or not
    """
    
    Y_pred = regressor.predict(X_test)
    Y_forecasted = regressor.predict(X1)
    
    """
    After we have tested our model on test data we are now in position to 
    show some visualization of our model.For this purpose we are going to use  python library matplotlib
    """
    
    #Visualising the Training set results
    print("\n")
    print(color.BOLD+"Global Consumption Forecasting"+color.END)
    print("\n")
    
    """
    Line graph representation
    """
    plt.scatter(X_train,Y_train,color="red")
    plt.plot(X_train,regressor.predict(X_train),color="blue")
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
    X_train_list =[]
    for x in X_train.flat:
        X_train_list.append(x)
    Y_train_predicted = regressor.predict(X_train)
    Y_train_list = []
    for y in Y_train_predicted.flat:
        Y_train_list.append(y)
    #print(X_train_list)
    #print(Y_train_list)
    plt.scatter(X_train,Y_train,color="red")
    plt.bar(X_train_list,Y_train_list)
    plt.title("Year vs Consumption (Training set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
        
    #plt.bar(list(X_train),list(regressor.predict(X_train)),label="Bar graph")
    #plt.show()
    
    
    
    #Visualising the test set result
    """
        Tabular form
    """
    X_value_list = []
    for x in X_test.flat:
        X_value_list.append(x)
    Y_actual_list = []
    for y in Y_test.flat:
        Y_actual_list.append(y)
    Y_predicted_list=[]
    for y in Y_pred.flat:
        Y_predicted_list.append(y)
# =============================================================================
#     print(X_value_list)
#     print(Y_actual_list)
#     print(Y_predicted_list)
# =============================================================================
    actual_list=[]
    predicted_list=[]
    counter=0
    for x in X_value_list:
        actual_list.append((x,math.ceil(Y_actual_list[counter])))
        counter=counter+1
    counter=0
    for x in X_value_list:
        predicted_list.append((x,math.ceil(Y_predicted_list[counter])))
        counter=counter+1
    print("Actual Global oil Consumption Value")
    print(actual_list)
    print("\n")
    print("Predicted Global oil Consumption Value")
    print(predicted_list)
    print("\n")
        
    """
    Line graph representation
    """
    
    plt.scatter(X_test,Y_test,color="red")
    plt.plot(X_test,Y_pred,color="blue")
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
    
    X_test_list =[]
    for x in X_test.flat:
        X_test_list.append(x)
    Y_test_predicted = regressor.predict(X_test)
    Y_test_list = []
    for y in Y_pred.flat:
        Y_test_list.append(y)
    #print(X_train_list)
    #print(Y_train_list)
    plt.scatter(X_test,Y_test,color="red")
    plt.bar(X_test_list,Y_test_list)
    plt.title("Year vs Consumption (Test set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    #Forecasting the result for next five years
    
    """
    Line graph representation
    """
    
    plt.scatter(X1,Y1,color="red")
    plt.plot(X1,regressor.predict(X1),color="blue")
    plt.title("Year vs Consumption (Forecasting)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(1)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    """
    Bar graph representation
    """
    
    X_forecast_list =[]
#    print(X1)
    for x in X1.flat:
        X_forecast_list.append(x)
    Y_test_predicted = regressor.predict(X_test)
    Y_forecast_list = []
    for y in Y_forecasted.flat:
        Y_forecast_list.append(y)
    #print(X_forecast_list)
    #print(Y_forecast_list)
    print(X_forecast_list)
    print(Y_forecast_list)
    plt.scatter(X1,Y1,color="red")
    plt.bar(X_forecast_list,Y_forecast_list)
    plt.title("Year vs Consumption (Forecasting)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(1)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()

"""
Function for predicting oil consumption in India
"""

def indian_forcast():
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
    
    X_train,X_test_India,Y_train,Y_test_India = train_test_split(X,Y,test_size = 0.2,random_state = 42)
    
    """
    In this section we are going to fit the simple regression model to our training set
    For this purpose we are using Linear Regression module inbuild in sklearn library
    
    """
    
    regressor = LinearRegression()
    regressor.fit(X_train,Y_train)
    
    """
    After training our model on training data set we need to 
    test it on test_data set to see whether it is efficient or not
    """
    
    Y_pred_India = regressor.predict(X_test_India)
    Y_forecasted = regressor.predict(X1)
    
    """
    After we have tested our model on test data we are now in position to 
    show some visualization of our model.For this purpose we are going to use  python library matplotlib
    """
    
    #Visualising the Training set results
    print("\n")
    print(color.BOLD+"Oil Consumption Forecasting in India"+color.END)
    print("\n")
    
    """
    Line graph representation
    """
    plt.scatter(X_train,Y_train,color="red")
    plt.plot(X_train,regressor.predict(X_train),color="blue")
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
    X_train_list =[]
    for x in X_train.flat:
        X_train_list.append(x)
    Y_train_predicted = regressor.predict(X_train)
    Y_train_list = []
    for y in Y_train_predicted.flat:
        Y_train_list.append(y)
    #print(X_train_list)
    #print(Y_train_list)
    plt.scatter(X_train,Y_train,color="red")
    plt.bar(X_train_list,Y_train_list)
    plt.title("Year vs Consumption (Training set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
        
    #plt.bar(list(X_train),list(regressor.predict(X_train)),label="Bar graph")
    #plt.show()
    
    """
        Tabular form
    """
    X_value_list = []
    for x in X_test_India.flat:
        X_value_list.append(x)
    Y_actual_list = []
    for y in Y_test_India.flat:
        Y_actual_list.append(y)
    Y_predicted_list=[]
    for y in Y_pred_India.flat:
        Y_predicted_list.append(y)
# =============================================================================
#     print(X_value_list)
#     print(Y_actual_list)
#     print(Y_predicted_list)
# =============================================================================
    actual_list=[]
    predicted_list=[]
    counter=0
    for x in X_value_list:
        actual_list.append((x,math.ceil(Y_actual_list[counter])))
        counter=counter+1
    counter=0
    for x in X_value_list:
        predicted_list.append((x,math.ceil(Y_predicted_list[counter])))
        counter=counter+1
    print("Actual oil Consumption Value in India")
    print(actual_list)
    print("\n")
    print("Predicted oil Consumption Value in India")
    print(predicted_list)
    print("\n")
        
    
    
    
    #Visualising the test set result
    """
    Line graph representation
    """
    
    plt.scatter(X_test_India,Y_test_India,color="red")
    plt.plot(X_test_India,Y_pred_India,color="blue")
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
    
    X_test_list =[]
    for x in X_test_India.flat:
        X_test_list.append(x)
    Y_test_predicted = regressor.predict(X_test_India)
    Y_test_list = []
    for y in Y_pred_India.flat:
        Y_test_list.append(y)
    #print(X_train_list)
    #print(Y_train_list)
    plt.scatter(X_test_India,Y_test_India,color="red")
    plt.bar(X_test_list,Y_test_list)
    plt.title("Year vs Consumption (Test set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    #Forecasting the result for next ten years
    
    """
    Line graph representation
    """
    
    plt.scatter(X1,Y1,color="red")
    plt.plot(X1,regressor.predict(X1),color="blue")
    plt.title("Year vs Consumption (Forecasting)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(1)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    """
    Bar graph representation
    """
    
    X_forecast_list =[]
    for x in X1.flat:
        X_forecast_list.append(x)
    Y_test_predicted = regressor.predict(X_test_India)
    Y_forecast_list = []
    for y in Y_forecasted.flat:
        Y_forecast_list.append(y)
    print(X_forecast_list)
    print(Y_forecast_list)
    #print(X_forecast_list)
    #print(Y_forecast_list)
    plt.scatter(X1,Y1,color="red")
    plt.bar(X_forecast_list,Y_forecast_list)
    plt.title("Year vs Consumption (Forecasting)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(1)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
"""
Function for predicting oil consumption in China

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
    
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 42)
    
    """
    In this section we are going to fit the simple regression model to our training set
    For this purpose we are using Linear Regression module inbuild in sklearn library
    
    """
    
    regressor = LinearRegression()
    regressor.fit(X_train,Y_train)
    
    """
    After training our model on training data set we need to 
    test it on test_data set to see whether it is efficient or not
    """
    
    Y_pred = regressor.predict(X_test)
    Y_forecasted = regressor.predict(X1)
    
    """
    After we have tested our model on test data we are now in position to 
    show some visualization of our model.For this purpose we are going to use  python library matplotlib
    """
    
    #Visualising the Training set results
    print("\n")
    print(color.BOLD+"Oil Consumption Forecasting in China"+color.END)
    print("\n")
    
    """
    Line graph representation
    """
    plt.scatter(X_train,Y_train,color="red")
    plt.plot(X_train,regressor.predict(X_train),color="blue")
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
    X_train_list =[]
    for x in X_train.flat:
        X_train_list.append(x)
    Y_train_predicted = regressor.predict(X_train)
    Y_train_list = []
    for y in Y_train_predicted.flat:
        Y_train_list.append(y)
    #print(X_train_list)
    #print(Y_train_list)
    plt.scatter(X_train,Y_train,color="red")
    plt.bar(X_train_list,Y_train_list)
    plt.title("Year vs Consumption (Training set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
        
    #plt.bar(list(X_train),list(regressor.predict(X_train)),label="Bar graph")
    #plt.show()
    
    
    
    #Visualising the test set result
    """
        Tabular form
    """
    X_value_list = []
    for x in X_test.flat:
        X_value_list.append(x)
    Y_actual_list = []
    for y in Y_test.flat:
        Y_actual_list.append(y)
    Y_predicted_list=[]
    for y in Y_pred.flat:
        Y_predicted_list.append(y)
# =============================================================================
#     print(X_value_list)
#     print(Y_actual_list)
#     print(Y_predicted_list)
# =============================================================================
    actual_list=[]
    predicted_list=[]
    counter=0
    for x in X_value_list:
        actual_list.append((x,math.ceil(Y_actual_list[counter])))
        counter=counter+1
    counter=0
    for x in X_value_list:
        predicted_list.append((x,math.ceil(Y_predicted_list[counter])))
        counter=counter+1
    print("Actual oil Consumption Value in China")
    print(actual_list)
    print("\n")
    print("Predicted Global oil Consumption in China")
    print(predicted_list)
    print("\n")
        
    """
    Line graph representation
    """
    
    plt.scatter(X_test,Y_test,color="red")
    plt.plot(X_test,Y_pred,color="blue")
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
    
    X_test_list =[]
    for x in X_test.flat:
        X_test_list.append(x)
    Y_test_predicted = regressor.predict(X_test)
    Y_test_list = []
    for y in Y_pred.flat:
        Y_test_list.append(y)
    #print(X_train_list)
    #print(Y_train_list)
    plt.scatter(X_test,Y_test,color="red")
    plt.bar(X_test_list,Y_test_list)
    plt.title("Year vs Consumption (Test set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    #Forecasting the result for next ten years
    
    """
    Line graph representation
    """
    
    plt.scatter(X1,Y1,color="red")
    plt.plot(X1,regressor.predict(X1),color="blue")
    plt.title("Year vs Consumption (Forecasting)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(1)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    """
    Bar graph representation
    """
    
    X_forecast_list =[]
    for x in X1.flat:
        X_forecast_list.append(x)
    Y_test_predicted = regressor.predict(X_test)
    Y_forecast_list = []
    for y in Y_forecasted.flat:
        Y_forecast_list.append(y)
    #print(X_forecast_list)
    #print(Y_forecast_list)
    print(Y_forecast_list)
    plt.scatter(X1,Y1,color="red")
    plt.bar(X_forecast_list,Y_forecast_list)
    plt.title("Year vs Consumption (Forecasting)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(1)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
"""
Function for predicting oil consumption in United States of America

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
    
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 42)
    
    """
    In this section we are going to fit the simple regression model to our training set
    For this purpose we are using Linear Regression module inbuild in sklearn library
    
    """
    
    regressor = LinearRegression()
    regressor.fit(X_train,Y_train)
    
    """
    After training our model on training data set we need to 
    test it on test_data set to see whether it is efficient or not
    """
    
    Y_pred = regressor.predict(X_test)
    Y_forecasted = regressor.predict(X1)
    
    """
    After we have tested our model on test data we are now in position to 
    show some visualization of our model.For this purpose we are going to use  python library matplotlib
    """
    
    #Visualising the Training set results
    print("\n")
    print(color.BOLD+"Oil Consumption Forecasting in United States of America"+color.END)
    print("\n")
    
    """
    Line graph representation
    """
    plt.scatter(X_train,Y_train,color="red")
    plt.plot(X_train,regressor.predict(X_train),color="blue")
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
    X_train_list =[]
    for x in X_train.flat:
        X_train_list.append(x)
    Y_train_predicted = regressor.predict(X_train)
    Y_train_list = []
    for y in Y_train_predicted.flat:
        Y_train_list.append(y)
    #print(X_train_list)
    #print(Y_train_list)
    plt.scatter(X_train,Y_train,color="red")
    plt.bar(X_train_list,Y_train_list)
    plt.title("Year vs Consumption (Training set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
        
    #plt.bar(list(X_train),list(regressor.predict(X_train)),label="Bar graph")
    #plt.show()
    
    
    
    #Visualising the test set result
    """
        Tabular form
    """
    X_value_list = []
    for x in X_test.flat:
        X_value_list.append(x)
    Y_actual_list = []
    for y in Y_test.flat:
        Y_actual_list.append(y)
    Y_predicted_list=[]
    for y in Y_pred.flat:
        Y_predicted_list.append(y)
# =============================================================================
#     print(X_value_list)
#     print(Y_actual_list)
#     print(Y_predicted_list)
# =============================================================================
    actual_list=[]
    predicted_list=[]
    counter=0
    for x in X_value_list:
        actual_list.append((x,math.ceil(Y_actual_list[counter])))
        counter=counter+1
    counter=0
    for x in X_value_list:
        predicted_list.append((x,math.ceil(Y_predicted_list[counter])))
        counter=counter+1
    print("Actual oil Consumption Value in USA")
    print(actual_list)
    print("\n")
    print("Predicted Global oil Consumption in USA")
    print(predicted_list)
    print("\n")
    """
    Line graph representation
    """
    
    plt.scatter(X_test,Y_test,color="red")
    plt.plot(X_test,Y_pred,color="blue")
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
    
    X_test_list =[]
    for x in X_test.flat:
        X_test_list.append(x)
    Y_test_predicted = regressor.predict(X_test)
    Y_test_list = []
    for y in Y_pred.flat:
        Y_test_list.append(y)
    #print(X_train_list)
    #print(Y_train_list)
    plt.scatter(X_test,Y_test,color="red")
    plt.bar(X_test_list,Y_test_list)
    plt.title("Year vs Consumption (Test set)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
     #Forecasting the result for next ten years
    
    """
    Line graph representation
    """
    
    plt.scatter(X1,Y1,color="red")
    plt.plot(X1,regressor.predict(X1),color="blue")
    plt.title("Year vs Consumption (Forecasting)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(1)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
    
    """
    Bar graph representation
    """
    
    X_forecast_list =[]
    for x in X1.flat:
        X_forecast_list.append(x)
    Y_test_predicted = regressor.predict(X_test)
    Y_forecast_list = []
    for y in Y_forecasted.flat:
        Y_forecast_list.append(y)
    #print(X_forecast_list)
    #print(Y_forecast_list)
    print(Y_forecast_list)
    plt.scatter(X1,Y1,color="red")
    plt.bar(X_forecast_list,Y_forecast_list)
    plt.title("Year vs Consumption (Forecasting)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(1)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
if __name__ =='__main__':
    global_forcast()
    indian_forcast()
    china_forcast()
    USA_forcast()
    





