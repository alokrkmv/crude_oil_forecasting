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

dataset = pd.read_csv("data_formatted.csv")
forecasting_dataset = pd.read_csv("forecasting.csv")
"""
We are creating matrix of independent variable and vector of dependent variable
"""
X = dataset.iloc[:,0:1].values
Y = dataset.iloc[:,1:2].values
X1 = forecasting_dataset.iloc[:,0:1].values
Y1 = forecasting_dataset.iloc[:,1:2].values
# Y = dataset.iloc[:,3].values
"""
Here we will check for exitance of any missing value and replace that missing value by mean of the 
column
"""
from sklearn.preprocessing import Imputer
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
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 42)
