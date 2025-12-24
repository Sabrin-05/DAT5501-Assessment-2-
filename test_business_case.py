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