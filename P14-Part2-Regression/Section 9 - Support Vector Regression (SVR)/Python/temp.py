# -*- coding: utf-8 -*-
"""
Created on Fri May 29 18:37:54 2020

@author: $ahil
"""


# Support Vector Regression (SVR)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 2:2].values