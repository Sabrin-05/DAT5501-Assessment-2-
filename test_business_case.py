# import necessary libraries

import unittest 
import numpy as np
from business_case import clean_co2_data
from business_case import clean_electricity_data
from business_case import merge_datasets
from business_case import rename_coloumns
from business_case import optimise_k_means
import pandas as pd
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
    

    @patch("business_case.plt.show")   # prevents plot window from opening
    @patch("business_case.KMeans.fit")

    def test_opimise_k_means(self):
        '''
        Function: Checks weather optimise_k_means functions is 
        correctly calculating the optimal number of clusters
        '''
        
        
        # Create a simple numeric dataframe
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 3, 4, 5, 6]
        })

        # Run the function
        optimise_k_means(df, max_k=5)

        # Check that KMeans.fit was called 4 times (k = 1 to 4)
        self.assertEqual(mock_fit.call_count, 4)

        # Check that plt.show() was called once
        mock_show.assert_called_once()

if __name__ == "__main__":
    unittest.main()



