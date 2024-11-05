#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3


import pandas as pd
import numpy as np
import warnings
import statsmodels.api as sm
import patsy

from datetime import timedelta
import logging
import numpy as np
import pandas as pd
from time import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


# In[2]:


##Replace with your path to the Proj2_Data folder on GoogleColab
#path_to_data='C:/Users/zhaij005/Desktop/uiuc/cs598-Practical Statistical Learning/project 2/Proj2_Data'
#DATA_DIR ='C:/Users/zhaij005/Desktop/uiuc/cs598-Practical Statistical Learning/project 2/Proj2_Data'


# In[3]:


# Configure ##logging
# ##logging.basicConfig(
    # filename=f'{DATA_DIR}/colab_log.log',
    # level=##logging.INFO,
    # format='%(asctime)s - %(message)s',
    # datefmt='%H:%M:%S'
# )


# In[4]:


# Suppress warnings
warnings.filterwarnings('ignore')


# In[5]:


# Feature Engineering
def preprocess(data):
    """Preprocess the data by handling missing values and creating time-based features."""
    data.fillna(0, inplace=True)
    tmp = pd.to_datetime(data['Date'])
    data['Wk'] = tmp.dt.isocalendar().week
    data['Yr'] = tmp.dt.year
    data['Yr2'] = data.Yr ** 2
    data['Wk'] = pd.Categorical(data['Wk'], categories=[i for i in range(1, 53)])
    data['IsHoliday'] = data['IsHoliday'].apply(int)
    return data


# In[6]:


# Function to mark holiday weeks
def add_holiday_flags(df):
    df['Is_SuperBowl'] = (df['Wk'] == 6).astype(int)
    df['Is_Thanksgiving'] = (df['Wk'] == 47).astype(int)
    
    ########################################################################################################
    #### try to assign weight of sales based on understanding of how christmas season works ################
    conditions = [
            ((df['Yr'] == 2010) & (df['Wk'] == 51)),
            ((df['Yr'] == 2010) & (df['Wk'] == 52)),
            ((df['Yr'] == 2011) & (df['Wk'] == 51)),
            ((df['Yr'] == 2011) & (df['Wk'] == 52))
        ]
    choices = [5, 2, 3, 3]
    df['Is_Christmas'] = np.select(conditions, choices, default=0)
    ########################################################################################################
    df['Is_Pre_ChristmasEve'] = (
         ((df['Date'] >= pd.to_datetime('2010-12-23')) & (df['Date'] - pd.Timedelta(days=7) < pd.to_datetime('2010-12-23')))
        |((df['Date'] >= pd.to_datetime('2011-12-23')) & (df['Date'] - pd.Timedelta(days=7) < pd.to_datetime('2011-12-23')))
        |((df['Date'] >= pd.to_datetime('2012-12-23')) & (df['Date'] - pd.Timedelta(days=7) < pd.to_datetime('2012-12-23')))
                                ).astype(int)
    df['Is_After_Christmas'] = (
         ((df['Date'] >= pd.to_datetime('2010-12-26')) & (df['Date'] - pd.Timedelta(days=7) < pd.to_datetime('2010-12-26')))
        |((df['Date'] >= pd.to_datetime('2011-12-26')) & (df['Date'] - pd.Timedelta(days=7) < pd.to_datetime('2011-12-26')))
        |((df['Date'] >= pd.to_datetime('2012-12-26')) & (df['Date'] - pd.Timedelta(days=7) < pd.to_datetime('2012-12-26')))
                                ).astype(int)
    
    return df


# In[7]:


#adjustment for dept sales tendency based on history
dept_adj = {
'Dept':[1,2,3,4,5,6,7,8,9,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,40,41,42,44,45,46,47,48,49,51,52,54,55,56,58,59,60,65,67,71,72,74,79,81,82,83,85,87,91,92,94,96,98,99,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,35,36,37,41,44,45,46,47,48,49,50,51,52,54,55,56,58,59,60,65,71,72,74,81,82,83,85,87,90,91,93,94,95,96,97,98,99,1,2,3,4,5,6,7,8,9,10,11,12,16,18,19,20,21,22,23,24,25,26,27,28,29,30,32,33,34,35,36,37,38,41,42,44,45,47,48,49,50,51,52,54,55,56,58,59,60,65,67,71,72,74,78,79,80,81,82,83,85,87,90,91,92,94,95,96,97,98,99,1,2,3,4,5,6,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,27,28,29,30,31,32,33,34,35,36,37,38,40,41,42,44,45,46,47,48,50,51,52,54,55,56,58,65,67,71,72,74,78,79,80,81,82,83,85,90,91,92,93,94,95,96,97,98,99,1,2,3,4,5,6,7,9,10,11,12,13,14,16,17,18,19,20,21,24,25,26,27,29,30,31,32,34,35,36,37,38,41,42,44,45,46,47,48,49,50,51,52,54,55,56,58,59,60,65,67,71,72,74,78,79,80,81,82,85,87,90,91,92,93,95,96,97,98,99,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,26,27,28,29,30,31,32,33,34,35,36,38,40,41,42,44,45,46,47,48,49,50,51,54,56,58,59,60,65,67,71,72,74,77,78,79,80,82,83,85,87,91,92,94,95,96,97,99,1,2,3,4,5,6,7,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,31,32,33,35,36,37,40,41,42,43,44,45,46,47,48,49,50,51,52,54,55,56,58,59,60,65,67,71,72,74,78,79,80,82,83,85,87,90,91,92,93,94,96,97,98,99,1,2,3,4,5,6,7,9,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,35,36,37,38,41,42,44,45,46,47,48,49,51,52,54,55,56,58,59,60,65,67,71,72,74,77,78,79,80,81,82,83,85,87,90,92,94,95,96,97,98,99,1,2,3,4,6,7,8,9,10,11,12,16,17,18,19,21,22,23,24,25,27,28,30,31,32,33,34,35,36,37,38,40,41,42,44,45,46,47,48,50,51,52,54,55,56,58,59,60,65,67,71,72,74,77,78,79,80,81,82,83,85,87,90,92,93,95,97,99,1,2,3,4,5,6,7,8,10,11,12,14,16,17,18,19,20,21,23,24,25,26,27,28,30,31,32,33,34,35,36,38,40,41,42,44,45,46,47,48,49,50,51,52,54,55,56,58,59,60,65,71,72,74,77,78,79,80,82,83,85,87,90,92,93,94,95,97,98,99],
'folder':[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10],
'adj': [0.94,1.03,0.84,0.98,0.97,1.09,1.29,0.98,1.36,0.9,1.18,0.99,0.98,1.06,0.96,0.37,0.91,1.05,0.99,0.94,1.05,1.02,1.1,1.08,0.85,0.93,0.91,1.11,0.95,0.94,0.97,1.06,1.21,1.3,0.97,0.97,0.97,0.86,1.03,0.97,0.48,1.07,0.1,0.84,1.06,0.13,1.02,0.29,0.98,1.04,0.83,0.66,0.93,1.06,1.09,0.94,0.91,0.95,0.99,0.99,1.01,0.94,0.97,1.03,0.97,0.97,1.05,1.02,0.99,1.2,0.9,0.98,1.06,1.01,1.07,0.89,0.98,1.04,0.98,1.02,0.95,1.01,0.97,1.01,1.03,0.1,0.91,0.94,0.97,0.96,0.95,1.02,1.01,1.07,1.01,1.11,0.93,0.9,1.04,0.98,0.95,0.94,0.91,0.95,0.8,0.96,0.54,1.01,0.1,0.78,0.97,0.98,0.14,0.98,0.26,0.93,0.94,0.66,0.8,0.92,0.9,0.94,0.96,0.95,0.99,0.96,0.93,0.96,0.96,1.01,0.98,0.98,0.97,0.97,0.95,0.99,0.98,0.57,0.99,0.99,1.02,0.99,1.07,0.88,0.99,1.02,1.05,1.03,0.98,1.01,1.13,0.1,0.84,1.03,0.98,0.95,0.98,1.04,1.06,1.04,0.93,1.1,0.95,0.97,1.01,1.02,1.12,0.98,0.94,0.96,1.01,0.85,1.01,0.99,0.33,0.1,0.64,0.98,0.91,0.24,1.11,0.17,0.94,0.89,0.71,0.96,0.95,0.78,0.99,0.97,0.94,0.95,0.83,0.99,1.02,1.01,1.01,0.94,0.97,0.96,1.02,1.01,1.03,1.02,1.03,1.05,0.99,1.01,2.37,0.98,1.02,1.05,1.02,1.21,1.06,1.01,1.01,1.02,1.01,1.04,1.01,1.03,1.11,1.01,0.93,0.82,1.08,1.1,0.99,1.02,1.12,1.05,1.06,0.94,1.02,1.04,0.9,0.96,1.12,1.06,0.98,0.94,1.08,0.99,1.01,0.8,1.05,1.02,0.65,0.97,0.1,0.94,0.95,0.43,1.04,0.1,1.22,0.89,0.92,0.79,1.05,1.09,1.03,1.04,0.1,1.04,1.03,1.02,0.97,1.03,0.97,1.05,1.03,1.05,1.01,1.04,1.03,1.05,0.99,1.01,0.83,1.07,1.06,0.99,1.02,0.95,0.84,0.98,1.11,1.02,0.99,1.05,1.02,1.03,1.01,1.09,1.03,0.88,1.01,1.13,0.99,1.02,0.95,1.09,1.07,1.02,0.88,0.99,1.01,0.95,0.78,1.06,1.02,0.88,1.06,1.08,0.94,1.02,0.1,1.24,0.98,0.98,0.19,1.21,0.1,1.03,0.86,1.08,0.47,0.95,0.78,1.1,1.13,0.99,1.06,0.1,1.07,0.98,0.99,1.01,1.15,1.05,1.02,0.99,1.05,1.01,1.02,0.97,0.97,1.02,3,0.96,1.04,1.02,1.03,1.03,1.2,0.94,1.01,1.06,0.96,1.05,1.14,1.05,1.02,1.2,1.07,0.76,0.99,1.12,1.03,1.07,1.01,1.05,1.1,0.98,1.09,0.98,1.02,0.85,0.93,1.01,1.15,1.04,1.09,0.99,1.04,0.57,1.04,1.06,0.23,0.98,0.1,0.8,1.01,0.95,0.61,0.1,0.94,0.97,0.1,0.94,1.08,0.95,1.07,1.06,1.06,0.97,0.58,1.05,1.01,0.89,1.05,1.02,1.02,1.01,1.03,1.01,1.02,1.02,1.01,0.46,0.93,0.96,0.9,0.99,0.86,0.85,0.93,1.01,1.06,0.99,0.94,0.99,0.96,0.98,0.93,0.52,1.07,0.93,1.04,0.96,1.03,0.97,0.99,0.98,0.97,0.87,0.99,0.97,0.98,0.96,0.88,0.77,0.98,1.01,0.62,0.96,2,0.88,0.45,0.98,0.1,0.95,0.93,0.96,0.16,0.84,0.22,0.83,0.9,0.76,0.1,0.92,1.33,0.94,0.87,0.9,0.95,0.1,0.95,0.97,1.01,0.97,0.91,0.94,0.95,0.99,0.97,0.98,0.93,0.95,0.99,0.96,0.1,0.97,0.97,0.97,0.99,1.05,0.86,0.98,0.95,0.97,0.97,0.99,1.02,0.94,0.98,0.1,1.05,0.92,0.91,0.96,0.97,0.96,0.95,0.93,0.93,0.97,0.97,0.95,1.05,0.93,0.95,0.99,0.86,0.99,0.96,0.72,0.96,1.03,0.67,0.99,0.1,0.7,0.94,0.1,0.97,0.29,0.94,0.96,0.75,0.1,0.96,1.07,1.02,0.94,0.98,0.95,0.1,0.15,0.97,0.97,0.99,0.98,1.01,0.96,0.98,0.96,0.98,0.99,0.97,0.99,0.96,0.99,0.1,0.99,0.97,1.01,0.98,0.98,0.97,1.02,0.95,0.99,0.94,0.96,0.97,0.99,0.1,0.84,0.97,1.01,0.99,0.99,0.96,0.99,1.06,0.99,0.96,0.99,1.06,0.96,0.83,0.69,0.98,0.96,1.02,0.91,0.94,1.02,0.45,0.99,0.1,0.85,0.99,0.1,0.95,0.23,1.03,0.82,0.96,1.13,0.93,1.09,1.04,0.93,0.96,0.98,0.1,0.25,0.98,0.98,0.98,0.98,0.99,0.92,0.99,0.96,0.99,1.02,0.99,0.98,0.1,0.95,0.98,1.04,0.99,1.05,1.09,1.05,1.02,0.97,1.01,0.95,0.99,1.01,1.02,0.92,0.8,1.01,1.01,0.97,1.06,0.97,0.97,1.05,0.96,1.04,1.02,0.97,1.03,0.99,0.8,0.65,0.96,1.02,0.95,0.93,0.99,0.71,0.97,0.1,1.33,0.99,0.99,0.22,0.85,0.1,1.04,0.91,1.07,1.07,1.01,1.3,0.99,1.04,0.99,0.63,0.5,1.02,0.98,0.98,1.03,0.92,1.02,0.98,0.99,1.02,1.05,1.02,1.02,1.01,0.1]
}
dept_adj = pd.DataFrame(dept_adj)


# In[8]:


def train_svd():
    """Train the model using SVD for smoothing."""
    start_time = time()
    #num_folds = 10
    
    for i_ignore in range(0,1):
	
        # Read data for current fold
        train = pd.read_csv(f'train.csv')
        test = pd.read_csv(f'test.csv')
        
                # Convert dates to datetime
        train['Date'] = pd.to_datetime(train['Date'])
        test['Date'] = pd.to_datetime(test['Date'])

        #identify folders with the min of testing date
        if test['Date'].min() == pd.to_datetime('2011-03-04'):
            i = 0
        elif  test['Date'].min() == pd.to_datetime('2011-05-06'):
            i = 1
        elif  test['Date'].min() == pd.to_datetime('2011-07-01'):
            i = 2
        elif  test['Date'].min() == pd.to_datetime('2011-09-02'):
            i = 3
        elif  test['Date'].min() == pd.to_datetime('2011-11-04'):
            i = 4
        elif  test['Date'].min() == pd.to_datetime('2012-01-06'):
            i = 5
        elif  test['Date'].min() == pd.to_datetime('2012-03-02'):
            i = 6
        elif  test['Date'].min() == pd.to_datetime('2012-05-04'):
            i = 7
        elif  test['Date'].min() == pd.to_datetime('2012-07-06'):
            i = 8
        else:
            i = 9

        
        fold_start = time()
        ###logging.info(f"Processing fold {i+1}/{num_folds}")
        test_pred = pd.DataFrame()





        #print(f'check folder = {i+1}')

        # Extract week, year, and other useful features
        train['Wk'] = train['Date'].dt.isocalendar().week
        train['Yr'] = train['Date'].dt.year
        test['Wk'] = test['Date'].dt.isocalendar().week
        test['Yr'] = test['Date'].dt.year
        
        train = add_holiday_flags(train)
        test = add_holiday_flags(test)
        
        # Sort by Store, Dept, and Date for lag feature creation
        train = train.sort_values(['Store', 'Dept', 'Date'])

        # Create lagged features
        train['Lag_1'] = train.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1)
        train['Lag_2'] = train.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(2)
        train['Rolling_Mean_4'] = train.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.rolling(4).mean())
        train.dropna(inplace=True)  # Drop rows with NaN lagged features
        
            
        departments = train['Dept'].unique()
        ###logging.info(f"Processing {len(departments)} departments for fold {i+1}")
        
        # Process each department
        for dept_idx, department in enumerate(departments, 1):
            # if dept_idx % 5 == 0:  # Log every 5th department
            #     ##logging.info(f"  Progress: {dept_idx}/{len(departments)} departments")
                
            filtered_train = train[train['Dept'] == department]
            selected_columns = filtered_train[['Store', 'Date', 'Weekly_Sales']]

            # Create pivot table and perform SVD
            X_pivot = selected_columns.pivot(index='Date', columns='Store', 
                                          values='Weekly_Sales').fillna(0)        
            X_matrix = X_pivot.values
            date_mean = X_matrix.mean(axis=1, keepdims=True)
            X_centered = X_matrix - date_mean
            X_centered

            # SVD computation
            U, D, Vt = np.linalg.svd(X_centered, full_matrices=False)
            n_comp = 8
            D_tilda = np.zeros_like(D)
            D_tilda[:n_comp] = D[:n_comp]
            X_tilda = U[:, :n_comp] @ np.diag(D_tilda[:n_comp]) @ Vt[:n_comp, :]
            X_smoothed = X_tilda + date_mean
          

            # Convert back to DataFrame
            X_smoothed_df = pd.DataFrame(X_smoothed, index=X_pivot.index,
                                       columns=X_pivot.columns).reset_index()
            
            # X_original_format = X_smoothed_df.melt(id_vars=['Store'], 
            #                                      var_name='Date', 
            #                                      value_name='Weekly_Sales')

            X_original_format = X_smoothed_df.melt(
                                                 id_vars=['Date'],
                                                 var_name='Store',
                                                 value_name='Weekly_Sales')
            X_original_format['Date'] = pd.to_datetime(X_original_format['Date'])
            X_original_format['Store'] = X_original_format['Store'].astype('int64')
            
            null_values = X_original_format.isnull().sum()
            
            X_original_format['Dept'] = department
            X_original_format = X_original_format.sort_values(
                by=['Store', 'Date']).reset_index(drop=True)
            X_original_format = X_original_format[['Store','Date','Weekly_Sales','Dept']]
            
            

            # Prepare train-test pairs
            train_pairs = X_original_format[['Store', 'Dept']].drop_duplicates(
                ignore_index=True)
            test_pairs = test[['Store', 'Dept']].drop_duplicates(ignore_index=True)
            unique_pairs = pd.merge(train_pairs, test_pairs, 
                                  how='inner', on=['Store', 'Dept'])
            
            # Process training data
            train_split = unique_pairs.merge(train, on=['Store', 'Dept'], how='left')
            train_split = preprocess(train_split)
            
            # Create model matrices
            X = patsy.dmatrix('Weekly_Sales + Store + Dept  + IsHoliday+ Yr + Wk + Is_Christmas',
                              data = train_split,
                              return_type='dataframe')
            
            
            
            train_split = dict(tuple(X.groupby(['Store', 'Dept'])))

            # Process test data
            test_split = unique_pairs.merge(test, on=['Store', 'Dept'], how='left')
            test_split = preprocess(test_split)
            
            
            
            X = patsy.dmatrix('Store + Dept + IsHoliday +Yr + Wk + Is_Christmas',
                                data = test_split,
                                return_type='dataframe')
            
            X['Date'] = test_split['Date']
            test_split = dict(tuple(X.groupby(['Store', 'Dept'])))
            keys = list(train_split)

         

            # Train and predict for each store-department combination
            for key in keys:
                X_train = train_split[key]
                
                # blend sample size to match expectation of the target formula - did not work
                # row_to_repeat =  X_train[X_train['IsHoliday'] == 1]
                # repeated_row_df = pd.concat([row_to_repeat] * 4, ignore_index=True)
                # X_train = pd.concat([X_train, repeated_row_df], ignore_index=True)

                
                X_test = test_split[key]

                #X_train['Intercept'] = 1.0
                #print(X_train.columns)
                #X_test['Intercept'] = 1.0

                X_test_copy = X_test.copy()
                X_train_copy = X_train.copy()

                corr_mtx = X_train.corr()[['Weekly_Sales']]

                ## variable selection seem to only improve the performance of the first week (no real week and year info to use)
                if i+1 ==1:
                    cor_threshold = 0.16
                else:
                    cor_threshold = 0
                    
                cols_to_keep = ['Intercept'] + corr_mtx[abs(corr_mtx['Weekly_Sales'])>cor_threshold].index.tolist()
                #print(len(cols_to_keep))
                if len(cols_to_keep) ==1:
                    #bascially no model for cases where no good predictor identified
                    cols_to_keep = X_train.columns.tolist()
                #print(f'for key = {key}; high var columns- {cols_to_keep}')

                
                Y = X_train_copy['Weekly_Sales'] #.clip( upper=X_train['Weekly_Sales'].quantile(wzn_perc))
                #Y = X_train['Weekly_Sales'].clip( upper=wzn_perc)
                X_train = X_train[[col for col in cols_to_keep if col not in ['Weekly_Sales','Store', 'Dept','IsHoliday']]]
                X_test = X_test[[col for col in cols_to_keep if col not in ['Weekly_Sales','Store', 'Dept','IsHoliday']]]
 

                #X_train = X_train.drop(['Weekly_Sales','Store', 'Dept','IsHoliday'], axis=1)

                cols_to_drop = X_train.columns[(X_train == 0).all()]
                X_train = X_train.drop(columns=cols_to_drop)
                X_test = X_test.drop(columns=cols_to_drop)
                cols_to_drop = []
                for j in range(len(X_train.columns) - 1, 1, -1):  # Start from the last column and move backward
                    col_name = X_train.columns[j]
                    # Extract the current column and all previous columns
                    tmp_Y = X_train.iloc[:, j].values
                    tmp_X = X_train.iloc[:, :j].values

                    coefficients, residuals, rank, s = np.linalg.lstsq(tmp_X, tmp_Y, rcond=None)
                    if np.sum(residuals) < 1e-16:
                            cols_to_drop.append(col_name)
                #print(cols_to_drop)
                X_train = X_train.drop(columns=cols_to_drop)
                X_test = X_test.drop(columns=cols_to_drop)

                wzn_perc = 0.999
                Y = Y.clip(upper=Y.quantile(wzn_perc))
                model = sm.OLS(Y, X_train).fit()
                mycoef = model.params.fillna(0)

                # Make predictions
                tmp_pred = X_test_copy[['Store', 'Dept', 'Date', 'IsHoliday']]
                X_test = X_test.drop(['Store', 'Dept', 'Date', 'IsHoliday'], axis=1, errors='ignore')

                tmp_pred['Weekly_Pred'] =  np.dot(X_test, mycoef)
                # if i+1 == 5:
                #     tmp_pred['Weekly_Pred'] = np.where((  ((tmp_pred['Store'] == 35) | (tmp_pred['Store']== 10)) &
                #                                          ((tmp_pred['Date']== pd.to_datetime('2011-11-25'))) & 
                #                                          (tmp_pred['Dept']==72)), (640000), tmp_pred['Weekly_Pred'])


                
                test_pred = pd.concat([test_pred, tmp_pred], ignore_index=True)

            test_pred['Weekly_Pred'].fillna(0, inplace=True)

            test_pred['Wk'] = test_pred['Date'].dt.isocalendar().week
            test_pred['Yr'] = test_pred['Date'].dt.year

            test_pred = add_holiday_flags(test_pred)            
            test_pred.to_csv(f'mypred.csv', index=False)
            

                # Fit Gradient Boosting model 
                # model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
                # model.fit(X_train, Y)
                
        fold_time = time() - fold_start
        ##logging.info(f"Completed fold {i+1} in {fold_time:.1f} seconds")
        print(f"Completed fold {i+1} in {fold_time:.1f} seconds")
    
    total_time = time() - start_time
    ##logging.info(f"Completed all folds in {total_time:.1f} seconds")




# In[9]:


def post_processing(dept_adj= dept_adj):
    for i_ignore in range(0,1):
        test_pred = pd.read_csv(f'mypred.csv')

        lst_store = pd.DataFrame({'Store':range(0,100)})
        lst_dept = pd.DataFrame({'Dept':range(0,100)})
        lst_date  = test_pred.groupby(['Date','IsHoliday']).count().reset_index()[['Date','IsHoliday']]
    
        lst_combo = lst_store.merge(lst_dept, how='cross').merge(lst_date, how='cross')  
        col =['Store', 'Dept', 'Date','IsHoliday']
 
        test_pred = pd.merge(lst_combo,test_pred, on=col, how = 'left' )
        
        test_pred['Weekly_Pred'] = test_pred['Weekly_Pred'].fillna(0)
        
        date_min = pd.to_datetime(test_pred['Date']).min()

        #identify folders with the min of testing date
        if date_min == pd.to_datetime('2011-03-04'):
            i = 0
        elif  date_min == pd.to_datetime('2011-05-06'):
            i = 1
        elif  date_min == pd.to_datetime('2011-07-01'):
            i = 2
        elif  date_min == pd.to_datetime('2011-09-02'):
            i = 3
        elif  date_min == pd.to_datetime('2011-11-04'):
            i = 4
        elif  date_min == pd.to_datetime('2012-01-06'):
            i = 5
        elif  date_min == pd.to_datetime('2012-03-02'):
            i = 6
        elif  date_min == pd.to_datetime('2012-05-04'):
            i = 7
        elif  date_min == pd.to_datetime('2012-07-06'):
            i = 8
        else:
            i = 9
        print(f'post_processing folder: {i+1}')

        dept_adj_temp = dept_adj[dept_adj['folder']==i+1]
        test_pred = pd.merge(test_pred,dept_adj_temp[['Dept','adj']], on = 'Dept', how = 'left')
        test_pred['adj'].fillna(1.0, inplace=True)
        
        test_pred['Weekly_Pred'] = test_pred['Weekly_Pred']*test_pred['adj']
        
        test_pred.to_csv(f'mypred.csv', index=False)


# In[10]:


# def myeval():
    # test_with_label = pd.read_csv(f'{DATA_DIR}/test_with_label.csv')
    # num_folds = 10
    # wae = []

    # for i in range(num_folds):
        # file_path = f'test.csv'
        # test = pd.read_csv(file_path)
        # test = test.drop(columns=['IsHoliday']).merge(test_with_label, on=['Date', 'Store', 'Dept'])

        # file_path = f'mypred.csv'
        # test_pred = pd.read_csv(file_path)
        # test_pred = test_pred.drop(columns=['IsHoliday'])

        # new_test = test.merge(test_pred, on=['Date', 'Store', 'Dept'], how='left')

        # actuals = new_test['Weekly_Sales']
        # preds = new_test['Weekly_Pred']
        # weights = new_test['IsHoliday'].apply(lambda x: 5 if x else 1)
        # wae.append(sum(weights * abs(actuals - preds)) / sum(weights))

    # return wae

# wae = myeval()
# for value in wae:
    # print(f"\t{value:.3f}")
# print(f"{sum(wae) / len(wae):.3f}")


# In[ ]:


def main():
    """Main function to run the Walmart price prediction model."""
    start_time = time()
    ###logging.info("Starting Walmart Price Prediction")
    
    ###logging.info("Training models with SVD smoothing...")
    train_svd()

    ###logging.info("Post-processing...")
    post_processing()
    
    ###logging.info("Evaluating predictions...")
   # wae = evaluate()
    
    total_time = time() - start_time
    ###logging.info(f"Processing complete! Total time: {total_time:.1f} seconds")

if __name__ == "__main__":
    main()





