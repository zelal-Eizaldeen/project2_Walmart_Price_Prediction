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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "/Users/zelal-ezaldeen/Documents/Zelal/MastersDegree/Y2/CS598psl/Assignments/Projects/project2/Data/Proj2_Data"

def preprocess(data):
    """Preprocess the data by handling missing values and creating time-based features."""
    data.fillna(0, inplace=True)

    tmp = pd.to_datetime(data['Date'])
    data['Wk'] = tmp.dt.isocalendar().week
    data['Yr'] = tmp.dt.year
    data['Wk'] = pd.Categorical(data['Wk'], categories=[i for i in range(1, 53)])
    data['IsHoliday'] = data['IsHoliday'].apply(int)
    return data

def train_svd():
    """Train the model using SVD for smoothing."""
    start_time = time()
    num_folds = 10
    
    for i in range(num_folds):
        fold_start = time()
        logging.info(f"Processing fold {i+1}/{num_folds}")
        test_pred = pd.DataFrame()

        # Read data for current fold
        train = pd.read_csv(f'{DATA_DIR}/fold_{i+1}/train.csv')
        test = pd.read_csv(f'{DATA_DIR}/fold_{i+1}/test.csv')

        # Define date ranges
        # start_last_year = pd.to_datetime(test['Date'].min()) - timedelta(days=375)
        # end_last_year = pd.to_datetime(test['Date'].max()) - timedelta(days=350)

        # Filter train data
        # tmp_train = train[(train['Date'] > str(start_last_year)) & 
        #                  (train['Date'] < str(end_last_year))].copy()

        departments = train['Dept'].unique()
        logging.info(f"Processing {len(departments)} departments for fold {i+1}")
        
        # Process each department
        for dept_idx, department in enumerate(departments, 1):
            if dept_idx % 5 == 0:  # Log every 5th department
                logging.info(f"  Progress: {dept_idx}/{len(departments)} departments")
                
            filtered_train = train[train['Dept'] == department]
            selected_columns = filtered_train[['Store', 'Date', 'Weekly_Sales']]

            # Create pivot table and perform SVD
            X_pivot = selected_columns.pivot(index='Store', columns='Date', 
                                          values='Weekly_Sales').fillna(0)
            X_matrix = X_pivot.values
            store_mean = X_matrix.mean(axis=1, keepdims=True)
            X_centered = X_matrix - store_mean

            # SVD computation
            U, D, Vt = np.linalg.svd(X_centered, full_matrices=False)
            n_comp = 8
            D_tilda = np.zeros_like(D)
            D_tilda[:n_comp] = D[:n_comp]
            X_tilda = U[:, :n_comp] @ np.diag(D_tilda[:n_comp]) @ Vt[:n_comp, :]
            X_smoothed = X_tilda + store_mean

            # Convert back to DataFrame
            X_smoothed_df = pd.DataFrame(X_smoothed, index=X_pivot.index, 
                                       columns=X_pivot.columns).reset_index()
            X_original_format = X_smoothed_df.melt(id_vars=['Store'], 
                                                 var_name='Date', 
                                                 value_name='Weekly_Sales')
            X_original_format['Dept'] = department
            X_original_format = X_original_format.sort_values(
                by=['Store', 'Date']).reset_index(drop=True)

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
            X = patsy.dmatrix('Weekly_Sales + Store + Dept  + IsHoliday+ Yr +I(Yr**2) + Wk',
                              data = train_split,
                              return_type='dataframe')
            
            
            train_split = dict(tuple(X.groupby(['Store', 'Dept'])))

            # Process test data
            test_split = unique_pairs.merge(test, on=['Store', 'Dept'], how='left')
            test_split = preprocess(test_split)
            
            X = patsy.dmatrix('Store + Dept + IsHoliday + Yr + I(Yr**2) + Wk',
                                data = test_split,
                                return_type='dataframe')
            
            X['Date'] = test_split['Date']
            test_split = dict(tuple(X.groupby(['Store', 'Dept'])))
            keys = list(train_split)


            # Train and predict for each store-department combination
            for key in keys:
                X_train = train_split[key]
                X_test = test_split[key]

                Y = X_train['Weekly_Sales']
                X_train = X_train.drop(['Weekly_Sales','Store', 'Dept','IsHoliday'], axis=1)

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
                X_train = X_train.drop(columns=cols_to_drop)
                X_test = X_test.drop(columns=cols_to_drop)
                
                model = sm.OLS(Y, X_train).fit()
                mycoef = model.params.fillna(0)

                # Make predictions
                tmp_pred = X_test[['Store', 'Dept', 'Date', 'IsHoliday']]
                X_test = X_test.drop(['Store', 'Dept', 'Date', 'IsHoliday'], axis=1, errors='ignore')

                tmp_pred['Weekly_Pred'] =  np.dot(X_test, mycoef)
                test_pred = pd.concat([test_pred, tmp_pred], ignore_index=True)

            test_pred['Weekly_Pred'].fillna(0, inplace=True)
            test_pred.to_csv(f'{DATA_DIR}/fold_{i+1}/mypred.csv', index=False)
            

                # Fit Gradient Boosting model 
                # model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
                # model.fit(X_train, Y)
                
        fold_time = time() - fold_start
        logging.info(f"Completed fold {i+1} in {fold_time:.1f} seconds")
    
    total_time = time() - start_time
    logging.info(f"Completed all folds in {total_time:.1f} seconds")

def evaluate():
    """Evaluate the model's predictions."""
    logging.info("Starting evaluation...")
    test_with_label = pd.read_csv(f'{DATA_DIR}/test_with_label.csv')
    num_folds = 10
    wae = []

    for i in range(num_folds):
        # Read test data and predictions
        test = pd.read_csv(f'{DATA_DIR}/fold_{i+1}/test.csv')
        test = test.merge(test_with_label, on=['Date', 'Store', 'Dept'])
        test_pred = pd.read_csv(f'{DATA_DIR}/fold_{i+1}/mypred.csv')
        
        # Merge and calculate weighted absolute error
        new_test = test.merge(test_pred, on=['Date', 'Store', 'Dept'], how='left')
        actuals = new_test['Weekly_Sales'].fillna(0)
        preds = new_test['Weekly_Pred'].fillna(0)
        weights = new_test['IsHoliday_x'].apply(lambda x: 5 if x else 1)
        
        wae_score = sum(weights * abs(actuals - preds)) / sum(weights)
        wae.append(wae_score)
        logging.info(f"Fold {i+1} WAE: {wae_score:.3f}")

    avg_wae = sum(wae) / len(wae)
    logging.info(f"Average WAE across all folds: {avg_wae:.3f}")
    
    return wae

def main():
    """Main function to run the Walmart price prediction model."""
    start_time = time()
    logging.info("Starting Walmart Price Prediction")
    
    logging.info("Training models with SVD smoothing...")
    train_svd()
    
    logging.info("Evaluating predictions...")
    wae = evaluate()
    
    total_time = time() - start_time
    logging.info(f"Processing complete! Total time: {total_time:.1f} seconds")

if __name__ == "__main__":
    main()
