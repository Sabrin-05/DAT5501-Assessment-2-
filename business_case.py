# import necessary libraries

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#~~~~~~~~~~~~~~~~~~~~~~~~
# load data and read csv
#~~~~~~~~~~~~~~~~~~~~~~~~

# read co2 csv dataset
co2_data = pd.read_csv('datasets/co2-emissions-per-capita.csv')
print(co2_data.head())  # display first few rows of the dataset
print(type(co2_data))

# read electricity csv dataset
electricity_data = pd.read_csv('datasets/electricity-fossil-renewables-nuclear-line.csv')
print(electricity_data.head())  # display first few rows of the dataset
print(type(electricity_data))


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


def clean_electricity_data(electricity_data):
    '''
    Function: cleans dataset by dropping columns, dropping NaN values, and filtering for specific years.
    
    Returns: cleaned dataframe
    '''

    # drop unnecessary columns
    electricity_data = electricity_data.drop(columns=['Code'])

    # filter for year 2000 onwards
    electricity_data = electricity_data[electricity_data['Year'] >= 2000]


    return electricity_data

electricity_data = clean_electricity_data(electricity_data)
print(electricity_data.head())  # display cleaned data

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# figure 1: Line plot, show the annual CO2 emissions per capita over the years
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

plt.figure(figsize=(10, 6))   

# top 10 co2 emitting countries
highest_emitters = co2_data.groupby('Entity')[
        'Annual CO₂ emissions (per capita)'].max().nlargest(10).index
print("Highest Emitters:", highest_emitters) # print highest emitting entities
co2_highest_emitters = co2_data[co2_data['Entity'].isin(highest_emitters)]


# Loop through top 10 co2 emitting countries and plot their data
for country in co2_highest_emitters['Entity'].unique():
    subset = co2_highest_emitters[co2_highest_emitters['Entity'] == country]

    # plot a separate line for each country
    plt.plot(
        subset['Year'],
        subset['Annual CO₂ emissions (per capita)'],
        label=country
    )

plt.legend()
plt.show()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Merging the datasets for K-Means Clustering plot
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def merge_datasets(co2_data, electricity_data, year='2022'):
    '''
    Function: Merges the emissions dataset witht the eclecricity dataset 
    To create one clean table with all necessary features for clustering 

    Parameters:
    -----------
    co2_data : pd.DataFrame
    CO2 emissions per capita (OWID)
    electricity_data : pd.DataFrame
    Electricity energy (OWID)
    year : int 
    Year to filter both datasets on 

    Returns:
    --------
    Merged dataset with desired features and no missing values
    '''


    # clean co2 emissions data
    co2_cleaned = (
        co2_data[co2_data["Year"] == year]
        .rename(columns={
            "Entity": "country",
            "Annual CO₂ emissions (per capita)": "co2_per_capita"
        })
        [["country", "co2_per_capita"]]
        .dropna()
    )

    # clean electricity mix data
    electricity_cleaned = (
        electricity_data[electricity_data["Year"] == year]
        .rename(columns={
            "Entity": "country",
            "Fossil fuels - % electricity": "fossil_share",
            "Renewables - % electricity": "renewable_share",
            "Nuclear - % electricity": "nuclear_share"
        })
        [["country", "fossil_share", "renewable_share", "nuclear_share"]]
        .dropna()
    )

    # debugging by checking number of rows and columns
    print("CO2 cleaned shape:", co2_cleaned.shape)
    print("Electricity cleaned shape:", electricity_cleaned.shape)
    
    # debugging by checking which year is in both datasets
    print(sorted(co2_data["Year"].unique()))
    print(sorted(electricity_data["Year"].unique()))

    # debugging by checking new column names exist
    print(co2_data.columns)
    print(electricity_data.columns)
    
    temp = co2_data[co2_data["Year"] == year].rename(columns={
    "Entity": "country",
    "Annual CO₂ emissions (per capita)": "co2_per_capita"
    })
    print(temp.head())


    # merge two datasets
    merged = co2_cleaned.merge(electricity_cleaned, on="country", how="inner")

    return merged


merged_df = merge_datasets(co2_data, electricity_data, year='2022')    
print(merged_df)
   
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Manually merging datasets togther as function doesnt work
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# AI assisst help create a loop to find countries from lines 175-189
co2_counts = co2_data.groupby("Year")["Entity"].nunique()
elec_counts = electricity_data.groupby("Year")["Entity"].nunique()

print(co2_counts.tail(20))
print(elec_counts.tail(20))

overlap = {}

for year in sorted(set(co2_data["Year"]).intersection(electricity_data["Year"])):
    co2_countries = set(co2_data[co2_data["Year"] == year]["Entity"])
    elec_countries = set(electricity_data[electricity_data["Year"] == year]["Entity"])
    overlap[year] = len(co2_countries.intersection(elec_countries))

# Print the top years
sorted(overlap.items(), key=lambda x: x[1], reverse=True)[:10]

# select desired year to merge data
co2_2022 = co2_data[co2_data["Year"] == 2022]
electricity_2022= electricity_data[electricity_data["Year"] == 2022]

# merge datasets on entities
joined_datasets = co2_2022.merge(electricity_2022, on="Entity", how="inner")
print(joined_datasets.head())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Transforming and Standerdising the data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
scaler = StandardScaler()

def optimise_k_means(data, max_k):
    '''
    Function: Works out the optimum number of clusters 

    Parameters:
    -----------

    '''
    means = []
    inertias =[]

    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)

        means.append(k)
        inertias.append(kmeans.inertia_)
    
    # generate the elbow plot 
    fig =plt.subplots(figsize=(10,5))
    plt.plot(means, inertias, '....')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()

    
