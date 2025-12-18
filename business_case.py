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

co2_data = pd.read_csv('datasets/co2-emissions-per-capita.csv')
print(co2_data.head())  # display first few rows of the dataset

electricity_data = pd.read_csv('datasets/electricity-fossil-renewables-nuclear-line.csv')
print(electricity_data.head())  # display first few rows of the dataset
