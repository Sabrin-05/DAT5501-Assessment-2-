# import necessary libraries

import unittest 
import numpy as np
from business_case import clean_co2_data
from business_case import clean_electricity_data
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

    
if __name__ == "__main__":
    unittest.main()



