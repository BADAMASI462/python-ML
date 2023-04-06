import numpy as np
import matplotlib.pyplot as plt
# MUST IMPORT PANDAS TO READ CSV FILES
import pandas as pd

# READING OF CSV FILES MUST BE DONE WITH A RAW STRING r IN ORDER TO READ \SLASH OTHER WISE USE /

dataset = pd.read_csv(r"C:\Users\badamasi\Downloads\Machine Learning-A-Z-Codes-Datasets\Machine Learning A-Z (Codes and Datasets)\Part 1 - Data Preprocessing\Section 2 -------------------- Part 1 - Data Preprocessing --------------------\Python\Data.csv")

#ILOC FUNCTION CALLED INDEX LOCATION TO READ INDEXES IN ROWS.
#THE RANGE IS USED TO READ ALL ROWS MEANING ALL FIELDS AS IN THIS CODE BELOW, 
# AND ALL COLUMNS EXCLUDING THE LAST ONE -1 THEN .VALUES MEANING THE VALUES.

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


print(x)
print(y)