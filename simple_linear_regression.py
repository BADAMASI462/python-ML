import numpy as np
import matplotlib.pyplot as plt
# MUST IMPORT PANDAS TO READ CSV FILES
import pandas as pd
# BEST LIBRARY USED WITH DATA SCIENCE
# WILL BE USED TO FILL IN MISSING DATA
from sklearn.impute import SimpleImputer # AN OBJECT OF THIS CLASS WILL BE CREATED TO FILL IN MISSING DATA
# LIBRARY TO ENCODE STRINGS(CATEGORICAL DATA). THE CLASS BELOW IS USED IN CONJUCTION WITH THE CLASS BELOW IT TO DO THE
# ONEHOTENCODER
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# LIBRARY TO ENCODE STRINGS(CATEGORICAL DATA). LABELENCODER
from sklearn.preprocessing import LabelEncoder
# LIBRARY TO SPLIT DATA INTO TRAIN AND TEST
from sklearn.model_selection import train_test_split
# LIBRARY TO USE LINEAR REGRESSION BOTH SIMPLE AND MULTIPLE REGRESSIONS
from sklearn.linear_model import LinearRegression



# READING OF CSV FILES MUST BE DONE WITH A RAW STRING r IN ORDER TO READ \SLASH OTHER WISE USE /

dataset = pd.read_csv(r"C:\Users\badamasi\Desktop\Machine Learning-A-Z-Codes-Datasets\Machine Learning A-Z (Codes and Datasets)\Part 2 - Regression\Section 4 - Simple Linear Regression\Python\Salary_Data.csv")

#ILOC FUNCTION CALLED INDEX LOCATION TO READ INDEXES IN ROWS.
#THE RANGE IS USED TO READ ALL ROWS MEANING ALL FIELDS AS IN THIS CODE BELOW, 
# AND ALL COLUMNS EXCLUDING THE LAST ONE -1 THEN .VALUES MEANING THE VALUES.

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


print(x)
print(y)


# SPLITTING DATASET INTO TRAIN AND TEST
# TEXT AND TRAIN VARIABLES MUST BE ARRANGED IN THIS WAY
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

# MODEL BUILDING
simpleLinearRegression = LinearRegression()
simpleLinearRegression.fit(x_train,y_train)
x_pred=simpleLinearRegression.predict(x)

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,simpleLinearRegression.predict(x_train
                                            ),color='blue'
)
plt.title('First graph'
          )
plt.xlabel('x'
           )
plt.xlabel('y'
           )
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,simpleLinearRegression.predict(x_train
                                            ),color='blue'
)
plt.title('predict graph'
          )
plt.xlabel('x'
           )
plt.xlabel('y'
           )
plt.show()

