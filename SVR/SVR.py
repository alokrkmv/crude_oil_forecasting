"""
Time series forecasting using SVR(Support vector Regression)

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
#    print(X1)
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
        As SVR library does not have inbuild feature scaling so we need to perfrom 
        the feature scaling manually in this case.    
    """
    print(X1)
    sc_X = StandardScaler()
    sc_Y = StandardScaler()
    sc_X1 = StandardScaler()
    sc_Y1 = StandardScaler()
    sc_X2 = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    Y_train = sc_Y.fit_transform(Y_train)
    X_test = sc_X1.fit_transform(X_test)
    Y_test = sc_Y1.fit_transform(Y_test)
    X1 = sc_X2.fit_transform(X1)
#    Y1 = sc_Y2.fit_transform(Y1)
    
    """
    In this section we are going to fit the SVR model to our training set
    For this purpose we are using SVM module inbuild in sklearn library
    
    """
    
    regressor = SVR(kernel = 'rbf')
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
    Y_train_transform = sc_Y.inverse_transform(Y_train)
    X_train_transform = sc_X.inverse_transform(X_train)
    plt.scatter(X_train_transform,Y_train_transform,color="red")
    plt.plot(X_train_transform,sc_Y.inverse_transform(regressor.predict(X_train)),color="blue")
    
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
    for x in X_train_transform.flat:
        X_train_list.append(x)
    Y_train_predicted_transform = sc_Y.inverse_transform(regressor.predict(X_train))
    Y_train_list = []
    for y in Y_train_predicted_transform.flat:
        Y_train_list.append(y)
    #print(X_train_list)
    #print(Y_train_list)
    plt.scatter(X_train_transform,Y_train_transform,color="red")
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
    Y_test_transform = sc_Y1.inverse_transform(Y_test)
    Y_pred_transform = sc_Y1.inverse_transform(Y_pred)
    X_test_transform = sc_X1.inverse_transform(X_test)
    print(X_test_transform)
    X_value_list = []
    for x in X_test_transform.flat:
        X_value_list.append(x)
    Y_actual_list = []
    for y in Y_test_transform.flat:
        Y_actual_list.append(y)
    Y_predicted_list=[]
    for y in Y_pred_transform.flat:
        Y_predicted_list.append(y)
    print(X_value_list)
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
    
    plt.scatter(X_test_transform,Y_test_transform,color="red")
    plt.plot(X_test_transform,Y_pred_transform,color="blue")
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
    for x in X_test_transform.flat:
        X_test_list.append(x)
    Y_test_predicted = regressor.predict(X_test)
    Y_test_list = []
    for y in Y_pred_transform.flat:
        Y_test_list.append(y)
    #print(X_train_list)
    #print(Y_train_list)
    plt.scatter(X_test_transform,Y_test_transform,color="red")
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
#    Y1_transform = sc_Y2.inverse_transform(Y1)
    Y1_pred_transform = sc_Y1.inverse_transform(regressor.predict(X1))
    X1_transform = sc_X2.inverse_transform(X1)
    print(X1_transform)
#    plt.scatter(X1,Y1,color="red")
    plt.plot(X1_transform, Y1_pred_transform,color="blue")
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
    for x in X1_transform.flat:
        X_forecast_list.append(x)
    Y_forecast_list = []
    for y in  Y1_pred_transform.flat:
        Y_forecast_list.append(y)
    print(X_forecast_list)
    print(Y_forecast_list)
#    plt.scatter(X1,Y1,color="red")
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
#    print(X1)
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
        As SVR library does not have inbuild feature scaling so we need to perfrom 
        the feature scaling manually in this case.    
    """
    print(X1)
    sc_X = StandardScaler()
    sc_Y = StandardScaler()
    sc_X1 = StandardScaler()
    sc_Y1 = StandardScaler()
    sc_X2 = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    Y_train = sc_Y.fit_transform(Y_train)
    X_test = sc_X1.fit_transform(X_test)
    Y_test = sc_Y1.fit_transform(Y_test)
    X1 = sc_X2.fit_transform(X1)
#    Y1 = sc_Y2.fit_transform(Y1)
    
    """
    In this section we are going to fit the SVR model to our training set
    For this purpose we are using SVM module inbuild in sklearn library
    
    """
    
    regressor = SVR(kernel = 'rbf')
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
    print(color.BOLD+"Oil Consumption Forecasting in India"+color.END)
    print("\n")
    
    """
    Line graph representation
    """
    Y_train_transform = sc_Y.inverse_transform(Y_train)
    X_train_transform = sc_X.inverse_transform(X_train)
    plt.scatter(X_train_transform,Y_train_transform,color="red")
    plt.plot(X_train_transform,sc_Y.inverse_transform(regressor.predict(X_train)),color="blue")
    
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
    for x in X_train_transform.flat:
        X_train_list.append(x)
    Y_train_predicted_transform = sc_Y.inverse_transform(regressor.predict(X_train))
    Y_train_list = []
    for y in Y_train_predicted_transform.flat:
        Y_train_list.append(y)
    #print(X_train_list)
    #print(Y_train_list)
    plt.scatter(X_train_transform,Y_train_transform,color="red")
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
    Y_test_transform = sc_Y1.inverse_transform(Y_test)
    Y_pred_transform = sc_Y1.inverse_transform(Y_pred)
    X_test_transform = sc_X1.inverse_transform(X_test)
    print(X_test_transform)
    X_value_list = []
    for x in X_test_transform.flat:
        X_value_list.append(x)
    Y_actual_list = []
    for y in Y_test_transform.flat:
        Y_actual_list.append(y)
    Y_predicted_list=[]
    for y in Y_pred_transform.flat:
        Y_predicted_list.append(y)
    print(X_value_list)
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
    print("Actual oil Consumption Value in India ")
    print(actual_list)
    print("\n")
    print("Predicted oil Consumption Value in India")
    print(predicted_list)
    print("\n")
        
    """
    Line graph representation
    """
    
    plt.scatter(X_test_transform,Y_test_transform,color="red")
    plt.plot(X_test_transform,Y_pred_transform,color="blue")
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
    for x in X_test_transform.flat:
        X_test_list.append(x)
    Y_test_predicted = regressor.predict(X_test)
    Y_test_list = []
    for y in Y_pred_transform.flat:
        Y_test_list.append(y)
    #print(X_train_list)
    #print(Y_train_list)
    plt.scatter(X_test_transform,Y_test_transform,color="red")
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
#    Y1_transform = sc_Y2.inverse_transform(Y1)
    Y1_pred_transform = sc_Y1.inverse_transform(regressor.predict(X1))
    X1_transform = sc_X2.inverse_transform(X1)
    print(X1_transform)
#    plt.scatter(X1,Y1,color="red")
    plt.plot(X1_transform, Y1_pred_transform,color="blue")
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
    for x in X1_transform.flat:
        X_forecast_list.append(x)
    Y_forecast_list = []
    for y in  Y1_pred_transform.flat:
        Y_forecast_list.append(y)
    #print(X_forecast_list)
    #print(Y_forecast_list)
#    plt.scatter(X1,Y1,color="red")
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
#    print(X1)
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
        As SVR library does not have inbuild feature scaling so we need to perfrom 
        the feature scaling manually in this case.    
    """
    print(X1)
    sc_X = StandardScaler()
    sc_Y = StandardScaler()
    sc_X1 = StandardScaler()
    sc_Y1 = StandardScaler()
    sc_X2 = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    Y_train = sc_Y.fit_transform(Y_train)
    X_test = sc_X1.fit_transform(X_test)
    Y_test = sc_Y1.fit_transform(Y_test)
    X1 = sc_X2.fit_transform(X1)
#    Y1 = sc_Y2.fit_transform(Y1)
    
    """
    In this section we are going to fit the SVR model to our training set
    For this purpose we are using SVM module inbuild in sklearn library
    
    """
    
    regressor = SVR(kernel = 'rbf')
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
    Y_train_transform = sc_Y.inverse_transform(Y_train)
    X_train_transform = sc_X.inverse_transform(X_train)
    plt.scatter(X_train_transform,Y_train_transform,color="red")
    plt.plot(X_train_transform,sc_Y.inverse_transform(regressor.predict(X_train)),color="blue")
    
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
    for x in X_train_transform.flat:
        X_train_list.append(x)
    Y_train_predicted_transform = sc_Y.inverse_transform(regressor.predict(X_train))
    Y_train_list = []
    for y in Y_train_predicted_transform.flat:
        Y_train_list.append(y)
    #print(X_train_list)
    #print(Y_train_list)
    plt.scatter(X_train_transform,Y_train_transform,color="red")
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
    Y_test_transform = sc_Y1.inverse_transform(Y_test)
    Y_pred_transform = sc_Y1.inverse_transform(Y_pred)
    X_test_transform = sc_X1.inverse_transform(X_test)
    print(X_test_transform)
    X_value_list = []
    for x in X_test_transform.flat:
        X_value_list.append(x)
    Y_actual_list = []
    for y in Y_test_transform.flat:
        Y_actual_list.append(y)
    Y_predicted_list=[]
    for y in Y_pred_transform.flat:
        Y_predicted_list.append(y)
    print(X_value_list)
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
    print("Predicted oil Consumption Value in China")
    print(predicted_list)
    print("\n")
        
    """
    Line graph representation
    """
    
    plt.scatter(X_test_transform,Y_test_transform,color="red")
    plt.plot(X_test_transform,Y_pred_transform,color="blue")
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
    for x in X_test_transform.flat:
        X_test_list.append(x)
    Y_test_predicted = regressor.predict(X_test)
    Y_test_list = []
    for y in Y_pred_transform.flat:
        Y_test_list.append(y)
    #print(X_train_list)
    #print(Y_train_list)
    plt.scatter(X_test_transform,Y_test_transform,color="red")
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
#    Y1_transform = sc_Y2.inverse_transform(Y1)
    Y1_pred_transform = sc_Y1.inverse_transform(regressor.predict(X1))
    X1_transform = sc_X2.inverse_transform(X1)
    print(X1_transform)
#    plt.scatter(X1,Y1,color="red")
    plt.plot(X1_transform, Y1_pred_transform,color="blue")
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
    for x in X1_transform.flat:
        X_forecast_list.append(x)
    Y_forecast_list = []
    for y in  Y1_pred_transform.flat:
        Y_forecast_list.append(y)
    #print(X_forecast_list)
    #print(Y_forecast_list)
#    plt.scatter(X1,Y1,color="red")
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
Function for prediciting oil consumption in USA
"""
def USA_forcast():
    dataset = pd.read_csv("dataset_formatted_2.csv")
    forecasting_dataset = pd.read_csv("forecasting.csv")
    """
    We are creating matrix of independent variable and vector of dependent variable
    """
    X = dataset.iloc[:,0:1].values
    Y = dataset.iloc[:,1:2].values
    X1 = forecasting_dataset.iloc[:,0:1].values
    Y1 = forecasting_dataset.iloc[:,1:2].values
#    print(X1)
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
        As SVR library does not have inbuild feature scaling so we need to perfrom 
        the feature scaling manually in this case.    
    """
    print(X1)
    sc_X = StandardScaler()
    sc_Y = StandardScaler()
    sc_X1 = StandardScaler()
    sc_Y1 = StandardScaler()
    sc_X2 = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    Y_train = sc_Y.fit_transform(Y_train)
    X_test = sc_X1.fit_transform(X_test)
    Y_test = sc_Y1.fit_transform(Y_test)
    X1 = sc_X2.fit_transform(X1)
#    Y1 = sc_Y2.fit_transform(Y1)
    
    """
    In this section we are going to fit the SVR model to our training set
    For this purpose we are using SVM module inbuild in sklearn library
    
    """
    
    regressor = SVR(kernel = 'rbf')
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
    print(color.BOLD+"Oil Consumption Forecasting in USA"+color.END)
    print("\n")
    
    """
    Line graph representation
    """
    Y_train_transform = sc_Y.inverse_transform(Y_train)
    X_train_transform = sc_X.inverse_transform(X_train)
    plt.scatter(X_train_transform,Y_train_transform,color="red")
    plt.plot(X_train_transform,sc_Y.inverse_transform(regressor.predict(X_train)),color="blue")
    
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
    for x in X_train_transform.flat:
        X_train_list.append(x)
    Y_train_predicted_transform = sc_Y.inverse_transform(regressor.predict(X_train))
    Y_train_list = []
    for y in Y_train_predicted_transform.flat:
        Y_train_list.append(y)
    #print(X_train_list)
    #print(Y_train_list)
    plt.scatter(X_train_transform,Y_train_transform,color="red")
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
    Y_test_transform = sc_Y1.inverse_transform(Y_test)
    Y_pred_transform = sc_Y1.inverse_transform(Y_pred)
    X_test_transform = sc_X1.inverse_transform(X_test)
    print(X_test_transform)
    X_value_list = []
    for x in X_test_transform.flat:
        X_value_list.append(x)
    Y_actual_list = []
    for y in Y_test_transform.flat:
        Y_actual_list.append(y)
    Y_predicted_list=[]
    for y in Y_pred_transform.flat:
        Y_predicted_list.append(y)
    print(X_value_list)
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
    print("Predicted oil Consumption Value in USA")
    print(predicted_list)
    print("\n")
        
    """
    Line graph representation
    """
    
    plt.scatter(X_test_transform,Y_test_transform,color="red")
    plt.plot(X_test_transform,Y_pred_transform,color="blue")
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
    for x in X_test_transform.flat:
        X_test_list.append(x)
    Y_test_predicted = regressor.predict(X_test)
    Y_test_list = []
    for y in Y_pred_transform.flat:
        Y_test_list.append(y)
    #print(X_train_list)
    #print(Y_train_list)
    plt.scatter(X_test_transform,Y_test_transform,color="red")
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
#    Y1_transform = sc_Y2.inverse_transform(Y1)
    Y1_pred_transform = sc_Y1.inverse_transform(regressor.predict(X1))
    X1_transform = sc_X2.inverse_transform(X1)
    print(X1_transform)
#    plt.scatter(X1,Y1,color="red")
    plt.plot(X1_transform, Y1_pred_transform,color="blue")
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
    for x in X1_transform.flat:
        X_forecast_list.append(x)
    Y_forecast_list = []
    for y in  Y1_pred_transform.flat:
        Y_forecast_list.append(y)
    #print(X_forecast_list)
    #print(Y_forecast_list)
#    plt.scatter(X1,Y1,color="red")
    plt.bar(X_forecast_list,Y_forecast_list)
    plt.title("Year vs Consumption (Forecasting)")
    plt.xlabel("Year of Consumption")
    plt.ylabel("Total Consumption")
    locator = matplotlib.ticker.MultipleLocator(1)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()
if __name__ == '__main__':
    global_forcast()
    india_forcast()
    china_forcast()
    USA_forcast()
    