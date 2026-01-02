# import necessary libraries

import unittest 
import numpy as np
from business_case import clean_co2_data, clean_electricity_data
import pandas as pd
from unittest.mock import patch


class my_unit_tests(unittest.TestCase):    
    def test_clean_co2_data(self):
        '''
        Function checks that the clean_co2_data function correctly cleans the CO2 dataset.
        
        '''
        
        # create a test dataframe to check weather the dataframe drops the 'Code' column correctly
        test_df = pd.DataFrame({
        'Entity': ['1','2'],
        'Code': ['000','111'],
        'Year': [1899,1900],
        'Result': [101,100]
        })

        cleansed = clean_co2_data(test_df)

        self.assertNotIn('Code', cleansed.columns) 
    
if __name__ == "__main__":
    unittest.main()



