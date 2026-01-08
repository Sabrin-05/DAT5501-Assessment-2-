#~~~~~~~~~~~~~~~~~~
# IMPORT LIBRARIES
#~~~~~~~~~~~~~~~~~~

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

#~~~~~~~~~~~~~~~~~~~~~
# FUNCTION DEFINITIONS
#~~~~~~~~~~~~~~~~~~~~~

def clean_co2_data(co2_data):
    '''
    Function: cleans dataset by dropping columns, dropping NaN values,
    and filtering for specific years.
    
    Returns: cleaned dataframe
    '''

    # drop unnecessary columns
    co2_data = co2_data.drop(columns=['Code'])

    # filter for year 2000 onwards
    co2_data = co2_data[co2_data['Year'] >= 2000]

    
    return co2_data


def clean_electricity_data(electricity_data):
    '''
    Function: cleans dataset by dropping columns, dropping NaN values,
    and filtering for specific years.
    
    Returns: cleaned dataframe
    '''

    # drop unnecessary columns
    electricity_data = electricity_data.drop(columns=['Code'])

    # filter for year 2000 onwards
    electricity_data = electricity_data[electricity_data['Year'] >= 2000]


    return electricity_data


def merge_datasets(co2_data, electricity_data):
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


    # filtered co2 emissions data on year 2022
    filtered_co2 = co2_data[co2_data['Year']== 2022]

    # filtered energy mix data on year 2022
    filtered_elec = electricity_data[electricity_data['Year'] == 2022]

    # merge two datasets
    merged = filtered_co2.merge(filtered_elec, on=["Entity","Year"], how="inner")

    return merged


def rename_coloumns(joined_datasets):
    '''
    Function: Renames the coloumns in the merged dataset for easier calling 
    
    Parameters:
    -----------
    Joined_datasets: 
    Containing co2 emissions dataset and electricity dataset merged

    Return:
    Newly named coloumns in dataframe
    '''

    data = joined_datasets.rename(columns={
            "Entity": "country",
            "Fossil fuels - % electricity": "fossil_share",
            "Renewables - % electricity": "renewable_share",
            "Nuclear - % electricity": "nuclear_share",
            "Annual CO₂ emissions (per capita)": "co2_per_capita"
            })
    
    return data 


def optimise_k_means(data, max_k):
    '''
    Function: Works out the optimum number of clusters 

    Parameters:
    -----------
    Data: 
    co2 emissions dataset 
    electricity mix dataset 

    Max_k:
    Maximum number of clusters that the algorithm will consider 

    Return:
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
    plt.plot(means, inertias, 'o-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()

#------------------------
# MAIN EXECUTION BLOCK
#------------------------

def main():
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

    #read income csv dataset
    income_data = pd.read_csv('datasets/world-bank-income-groups.csv')
    print(income_data.head())  # display first few rows of the dataset
    print(type(income_data))


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # data wrangling and cleaning
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    co2_data = clean_co2_data(co2_data)
    print(co2_data.head())  # display cleaned data


    electricity_data = clean_electricity_data(electricity_data)
    print(electricity_data.head())  # display cleaned data

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # figure 1: Line plot, show the annual CO2 emissions per capita over the years
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    plt.figure(figsize=(10, 6))   

    # top 10 co2 emitting countries
    highest_emitters = co2_data.groupby('Entity')[
            'Annual CO₂ emissions (per capita)'].max().nlargest(10).index
    print("Highest Emitters:", highest_emitters) 
    
    # print highest emitting entities
    co2_highest_emitters = co2_data[co2_data['Entity'].isin(highest_emitters)]

    # highlight Qatar, Aruba, Curacao
    bold_countries = {
        'Curacao': {'color': 'red', 'linewidth':3, 'linestyle':'--'},
        'Aruba': {'color': 'blue', 'linewidth':3, 'linestyle':'-.'},
        'Qatar': {'color': 'magenta', 'linewidth':3, 'linestyle':':'}
    }


    # Loop through top 10 co2 emitting countries and plot their data
    for country in co2_highest_emitters['Entity'].unique():
        subset = co2_highest_emitters[co2_highest_emitters['Entity'] == country]

        if country in bold_countries:
            style = bold_countries[country]
            plt.plot(
            subset['Year'],
            subset['Annual CO₂ emissions (per capita)'],
            label=country,
            color = style['color'],
            linewidth = style['linewidth'],
            linestyle = style['linestyle']
        )
        else:
            # plot a separate line for each country
            plt.plot(
                subset['Year'],
                subset['Annual CO₂ emissions (per capita)'],
                label=country,
                linewidth = 1,
                alpha = 0.6
            )

    plt.title('CO₂ Emissions per capita for top 10 emitting countries (2000-2023)')
    plt.xlabel('Year')
    plt.ylabel('CO₂ emissions (tonnes per capita)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("top_10_emitting_countries(2000-2022).png",
                 dpi=300,
                 bbox_inches='tight')
    plt.show()


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Merging the datasets for K-Means Clustering plot
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    merged_df = merge_datasets(co2_data, electricity_data)    
    print(merged_df)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Manually merging datasets togther as function doesnt work
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # group both datasets by year and entity
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


    # call the newly named dataframe
    data = rename_coloumns(joined_datasets)    
    print(data)

    # normalise the features ie each coloumn
    data[['fossil_share_T','renewable_share_T',
        'nuclear_share_T','co2_per_capita_T']] = scaler.fit_transform(data[['fossil_share','renewable_share',
        'nuclear_share','co2_per_capita']])

    print(data)


        
    optimise_k_means(data[['renewable_share_T',
                        'co2_per_capita_T']],10)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Figure 2: K-Means Cluster Plot
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(data[['renewable_share_T',
                    'co2_per_capita_T']])

    data['kmeans_3'] = kmeans.labels_
    print(data)

    # plotting the results

    plt.scatter(x=data['renewable_share'], y=data['co2_per_capita'], c=data['kmeans_3'])
    plt.xlim(-0.1,1)
    plt.ylim(3,1.5)
    plt.show()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Figure 3: Scatter plot with regression line 
    # Showing CO2 Emissions vs Fossil Fuel Share
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

    #Filter on 2019 for both datasets
    co2_2019 = co2_data[co2_data['Year'] == 2019]
    elec_2019 = electricity_data[electricity_data['Year'] == 2019]

    # merge both variables into one dataframe
    merged_2019 = co2_2019.merge(elec_2019, on=["Entity", "Year"], how="inner")

    # add income group dataset
    merged_2019 = merged_2019.merge(
        income_data[["Entity", "World Bank's income classification"]],
        on = 'Entity',
        how = 'left'
    )

    # colour code countries by income group
    income_colours = {
        'Low-income countries': '#1f77b4',
        'Lower-middle-income countries': '#2ca02c',
        'Upper-middle-income countries': '#ff7f0e',
        'High income countries': '#9467bd'
    }

    # find the outlier within the graphs
    merged_2019['co2_z'] = (merged_2019['Annual CO₂ emissions (per capita)'] - 
                            merged_2019['Annual CO₂ emissions (per capita)'].mean())/ merged_2019['Annual CO₂ emissions (per capita)'].std()

    outliers = merged_2019[merged_2019['co2_z'] > 2]   # countries more than 2 SD above mean
    print(outliers[['Entity', 'Fossil fuels - % electricity', 'Annual CO₂ emissions (per capita)']])


    # create and display the graph
    plt.figure(figsize=(10,6))

    for income_group, colour in income_colours.items():
        subset = merged_2019[
            (merged_2019["World Bank's income classification"] == income_group) &
            (merged_2019["co2_z"])
        ]


        plt.scatter(
            subset['Fossil fuels - % electricity'],
            subset['Annual CO₂ emissions (per capita)'],
            color = colour,
            s=40,
            alpha = 0.7,
            label = income_group
        )


    # anomalies
    plt.scatter(
        outliers['Fossil fuels - % electricity'],
        outliers['Annual CO₂ emissions (per capita)'],
        color='red',
        s=80,
        label='High-emission anomalies'
    )


    # print out the outlier countries
    for _, row in outliers.iterrows():
        plt.text(
            row['Fossil fuels - % electricity'] + 0.5,
            row['Annual CO₂ emissions (per capita)'] + 0.2,
            row['Entity'],
            fontsize=9
        )

    plt.xlabel('Fossil Fuel Share (%)')
    plt.ylabel('CO2 Emissions Per Capita (tonnes)')
    plt.title('CO₂ per capita vs Fossil Fuel Share by Income Group (2019)')
    plt.legend(title='Income Group')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("co2_emissions_vs_fossil_fuel_2019.png", dpi=300, bbox_inches='tight')
    plt.show()



    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Figure 4: Chloropleth Map showing CO2 emissions in 2022
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # read original co2 data
    df = pd.read_csv('datasets/co2-emissions-per-capita.csv')

    # filter co2 dataset for year 2022
    co2_2022 = df[df['Year'] == 2022]

    # rename 'Code' column as plotly expects 'Iso_code'
    co2_2022 = co2_2022.rename(columns={"Code": "iso_code"})

    # plot the chloropleth map 
    fig = px.choropleth(
        co2_2022,
        locations="iso_code",                     # ISO country codes
        color="Annual CO₂ emissions (per capita)",# Value to shade
        hover_name="Entity",                      # Hover label
        color_continuous_scale="Reds",            # Colour scheme
        title="CO₂ Emissions Per Capita (2022)"
    )

    fig.write_image(
        'heat_map_co2_emission_2022.png',
        width = 1000,
        height = 600,
        scale = 2
    )

    fig.show()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Figure 5: Stacked bar chart showing: Electricity Share composition by Income 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    income_entities = ['Low-income countries',
                       'Lower-middle-income countries',
                        'Upper-middle-income countries',
                        'High-income countries']
    
    
    energy_2019 = electricity_data[
        (electricity_data['Year'] == 2019) &
        (electricity_data['Entity'].isin(income_entities))
        ]
    
    energy_type = ['Nuclear - % electricity',
                   'Fossil fuels - % electricity',
                     'Renewables - % electricity']
    
    energy_2019 = energy_2019[['Entity'] + energy_type]

    # rename x- axis variables
    income_map = {
        'Low-income countries': 'Low-income',
        'Lower-middle-income countries': 'Lower-middle-income',
        'Upper-middle-income countries': 'Upper-middle-income',
        'High-income countries': 'High-income'
    }

    # settting entity as index
    energy_2019 = energy_2019.set_index('Entity')


    # create stacked bar chart
    ax = energy_2019.plot(
        kind = 'bar',
        stacked = True,
        figsize= (10,6)
    )

    ax.set_title('Energy Source Composition by Income (2019)')
    ax.set_xlabel('Income Group')
    ax.set_xticklabels(
        [income_map.get(label.get_text(), label.get_text())
         for label in ax.get_xticklabels()],
         rotation=0
    )
    ax.set_ylabel('Electricity Generation')
    ax.legend(
        title = 'Energy Source',
        bbox_to_anchor= (1.05,1),
        loc = 'upper left'
    )

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("energy_source_by_income_2019.png", dpi=300, bbox_inches='tight')
    plt.show()

    

    


#----------------------------
# RUN MAIN ONLY WHEN EXECUTED 
#----------------------------

if __name__ == '__main__':
    main()