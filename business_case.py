# import necessary libraries

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import tree 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

#~~~~~~~~~~~~~~~~~~~~~~~~
# load data and read csv
#~~~~~~~~~~~~~~~~~~~~~~~~

#
co2_data = pd.read_csv('datasets/co2-emissions-per-capita.csv')
print(co2_data.head())  # display first few rows of the dataset

#
electricity_data = pd.read_csv('datasets/electricity-fossil-renewables-nuclear-line.csv')
print(electricity_data.head())  # display first few rows of the dataset

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# data wrangling and cleaning
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def clean_co2_data(co2_data):
    '''
    Function: cleans dataset by dropping columns, dropping NaN values, and filtering for specific years.
    
    Returns: cleaned dataframe
    '''

    # drop unnecessary columns
    co2_data = co2_data.drop(columns=['Code'])

    # filter for year 2000 onwards
    co2_data = co2_data[co2_data['Year'] >= 2000]

    return co2_data

co2_data = clean_co2_data(co2_data)
print(co2_data.head())  # display cleaned data

