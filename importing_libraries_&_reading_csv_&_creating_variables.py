import numpy as np
import matplotlib.pyplot as plt
# MUST IMPORT PANDAS TO READ CSV FILES
import pandas as pd
# BEST LIBRARY USED WITH DATA SCIENCE
# WILL BE USED TO FILL IN MISSING DATA
from sklearn.impute import SimpleImputer # AN OBJECT OF THIS CLASS WILL BE CREATED TO FILL IN MISSING DATA

# READING OF CSV FILES MUST BE DONE WITH A RAW STRING r IN ORDER TO READ \SLASH OTHER WISE USE /

dataset = pd.read_csv(r"C:\Users\badamasi\Desktop\Machine Learning-A-Z-Codes-Datasets\Machine Learning A-Z (Codes and Datasets)\Part 1 - Data Preprocessing\Section 2 -------------------- Part 1 - Data Preprocessing --------------------\Python\Data.csv")

#ILOC FUNCTION CALLED INDEX LOCATION TO READ INDEXES IN ROWS.
#THE RANGE IS USED TO READ ALL ROWS MEANING ALL FIELDS AS IN THIS CODE BELOW, 
# AND ALL COLUMNS EXCLUDING THE LAST ONE -1 THEN .VALUES MEANING THE VALUES.

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


print(x)
print(y)

# OBJECT OF THIS CLASS simpleImputer() CREATED
imputer = SimpleImputer(missing_values=np.nan , strategy='mean')
# THIS METHOD WILL CHANGE THE MISSING VALUES OF THE GIVES COLUMNS BY DETECTING nan AND REPLACING
imputer.fit(x[:,1:3])
# THIS METHOD WILL RETURN THE VALUES TO THE VARIABLE. PRECISELY TO THE COLUMNS OF WHICH THERE ARE MISSING DATA
x[:,1:3] = imputer.transform(x[:,1:3])

print(x)