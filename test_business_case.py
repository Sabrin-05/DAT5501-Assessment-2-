# import necessary libraries

import unittest 
import numpy as np
from business_case import clean_co2_data
from business_case import clean_electricity_data
from business_case import merge_datasets
from business_case import rename_coloumns
from business_case import optimise_k_means
import pandas as pd
import plotly as px
from unittest.mock import patch


class TestCleanCO2Data(unittest.TestCase):   
    def test_clean_co2_data(self):
        '''
        Function checks that the clean_co2_data function correctly cleans the CO2 dataset.
        
        '''
        
        # create a test dataframe to check weather the dataframe drops the 'Code' column correctly
        test_df = pd.DataFrame({
        'Entity': ['1','2'],
        'Code': ['000','111'],
        'Year': [1899,2001],
        'Result': [101,100]
        })

        cleansed = clean_co2_data(test_df)

        self.assertNotIn('Code', cleansed.columns) 
        self.assertTrue((cleansed['Year'] >= 2000).all())

    
    def test_clean_electricity_data(self):
        '''
        Function checks that the clean_electricity_data function 
        correctly cleans the electricity dataset.
        
        '''

        # create a test dataframe to check weather the dataframe drops the 'Code' column correctly
        test_df = pd.DataFrame({
        'Entity': ['Costa Rica','Venezuela'],
        'Code': ['CR1','VN1'],
        'Year': [1999,2000],
        'Result': [90,110]
        })

        cleansed = clean_electricity_data(test_df)

        self.assertNotIn('Code', cleansed.columns) 
        self.assertTrue((cleansed['Year'] >= 2000).all())


    def test_rename_columns(self):
        '''
        Function: Checks where the rename columns function in main code is working correctly
        '''

        # create an example dataframe to mimic the original 
        test_df = pd.DataFrame({
        'Entity': ['Canada','Somalia'],
        'Fossil fuels - % electricity': [0.000987,0.7654567],
        'Renewables - % electricity': [0.456787654,0.21234],
        'Nuclear - % electricity': [0.0020123,1.09872],
        'Annual CO₂ emissions (per capita)': [0.9,0.007]
        })

        renamed = rename_coloumns(test_df)

        # check new coloumn names exist
        self.assertIn('country', renamed.columns)
        self.assertIn('fossil_share', renamed.columns)
        self.assertIn('renewable_share', renamed.columns)
        self.assertIn('nuclear_share', renamed.columns)
        self.assertIn('co2_per_capita', renamed.columns)

        # check old coloumns no longer exist 
        self.assertNotIn('Entity', renamed.columns)
        self.assertNotIn('Fossil fuels - % electricity', renamed.columns)
        self.assertNotIn('Renewables - % electricity', renamed.columns)
        self.assertNotIn('Nuclear - % electricity', renamed.columns)
        self.assertNotIn('Annual CO₂ emissions (per capita)', renamed.columns)

    
    def test_merge_datasets(self):
        '''
        Function: Checks wheather the co2 emissions dataset and the electricty dataset are
        merging correctly 
        '''

        # create an example dataframe to mimic the original co2 dataset
        test_co2_df = pd.DataFrame({
        'Entity': ['Australia', 'Peurto Rico','Qatar'],
        'Year': [2022,2021,2022],
        'Annual CO₂ emissions (per capita)': [0.007,0.9,0.15]
        })


        # create an example dataframe to mimic the original electricity share dataset
        test_elec_df = pd.DataFrame({
        'Entity': ['Australia','Qatar'],
        'Year': [2022,2022],
        'Fossil fuels - % electricity': [0.000987,0.99],
        'Renewables - % electricity': [0.456787654,0.23],
        'Nuclear - % electricity': [0.0020123,0.1987]
        })


        merged = merge_datasets(test_co2_df, test_elec_df)

        # ~~~~~~ Assertions ~~~~~~
    
        self.assertEqual(len(merged), 2)

        # check columns from both datasets exist in new dataframe
        self.assertIn('Entity', merged.columns)
        self.assertIn('Year', merged.columns)
        self.assertIn('Fossil fuels - % electricity', merged.columns)
        self.assertIn('Renewables - % electricity', merged.columns)
        self.assertIn('Nuclear - % electricity', merged.columns)
        self.assertIn('Annual CO₂ emissions (per capita)', merged.columns)
 
        # check only 2022 exists in dataframe
    


if __name__ == "__main__":
    unittest.main()



