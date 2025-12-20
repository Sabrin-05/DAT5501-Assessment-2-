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

    # top 10 co2 emitting countries
    highest_emitters = co2_data.groupby('Entity')[
        'Annual CO₂ emissions (per capita)'].max().nlargest(10).index
    print("Highest Emitters:", highest_emitters) # print highest emitting entities
    co2_data = co2_data[co2_data['Entity'].isin(highest_emitters)]

    return co2_data


co2_data = clean_co2_data(co2_data)
print(co2_data.head())  # display cleaned data



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# figure 1: show the annual CO2 emissions per capita over the years
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

plt.figure(figsize=(10, 6))   

# Loop through top 10 co2 emitting countries and plot their data
for country in co2_data['Entity'].unique():
    subset = co2_data[co2_data['Entity'] == country]

    # plot a separate line for each country
    plt.plot(
        subset['Year'],
        subset['Annual CO₂ emissions (per capita)'],
        label=country
    )

plt.legend()
plt.show()


