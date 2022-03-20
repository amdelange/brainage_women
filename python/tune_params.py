import csv
import pandas as pd
from functools import reduce
import numpy as np
import scipy
from scipy.stats.stats import pearsonr
from scipy import stats
import statsmodels.api as sm
import time
from numpy import mean
from numpy import std
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import xgboost as xgb
import matplotlib.pyplot as plt
import sys, os
import json


####################################

###### ADD PATHS TO DATA AND FOLDERS WHERE FILES WILL BE SAVED

datapath=""



# DICTIONARIES
dict = {}
corr = {}
pred = {}


#Output dictionary to store r and CI in json (to use for LaTeX table in step 8)
out_dict = {}

sex = ['both','male','female']

# DEFINE MODELS
model = ["global"]


for s in sex:
    for m in model:
        #saves the results to text files
        file=datapath+'Output_%s_%s_hyperpar.txt'% (m,s)
        with open(file, 'w') as text_file:

            text_file.write("===============\n")
            text_file.write("MODEL = %s\n" % m)
            text_file.write("===============\n")

            dict['%s' % m] = pd.read_csv(datapath+'%s_%s_hyperparam_tuning.csv' % (m,s))

            # CHECK FILE CONTENT AND LENGTH
            print ('printing head of %s for %s' % (m,s))
            print (dict['%s' % m].head(5))
            print ('printing number of columns %s for %s' % (m,s))
            print (len(dict['%s' % m].columns))
            print ('printing length of datafile %s for %s' % (m,s))
            print (len(dict['%s' % m]))


            # SPLIT THE FILE INTO X AND Y, WHERE X IS ALL THE MRI DATA AND Y IS AGE
            x = dict['%s' % m]
            print ('splitting data into x and y for %s' % m)
            y = x['Age']


            # MAKE A COPY OF THE DATA FRAME TO MERGE WITH ESTIMATED BRAIN AGE AT THE END OF SCRIPT
            x_copy = x.copy()


            # REMOVE VARIABLES FROM X THAT SHOULD NOT BE INCLUDED IN THE REGRESSOR
            x = x.drop('ID',1)
            x = x.drop('Age',1)


            # CHECK THAT X INCLUDES ONLY MRI VARIABLES, AND Y INCLUDES ONLY AGE
            print ('printing final x for %s' % m)
            print (x.head(5))
            print ('printing final y for %s' % m)
            print (y.head(5))


            # configure the nested cross-validation procedure
            cv_inner = KFold(n_splits=5, shuffle=True, random_state=1)
            cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)

            # define the model
            model = xgb.XGBRegressor(objective= 'reg:squarederror',nthread=4,seed=42)


            # define search space
            parameters = {'max_depth': range (2, 10, 1),
                        'n_estimators': range(60, 220, 40),
                        'learning_rate': [0.1, 0.01, 0.05]}

            # define search
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=parameters,
                scoring = 'neg_root_mean_squared_error',
                n_jobs = 4,
                cv = cv_inner,
                random_state=42,
                verbose = True,
                refit=True)

            result = search.fit(x, y)
            best_est = result.best_estimator_
            print("BEST ESTIMATOR")
            print(best_est)



            text_file.write ('validating %s model\n' % m)

            text_file.write ('------------------------------\n')
            text_file.write ('RMSE values:\n')
            RMSE = cross_val_score(search, x, y, cv=cv_outer,scoring='neg_root_mean_squared_error',n_jobs = 4)
            text_file.write('Mean and STD for RMSE: %.3f (%.3f)\n' % (mean(RMSE), std(RMSE)))

            text_file.write ('------------------------------\n')
            text_file.write ('MAE values:\n')
            MAE = cross_val_score(search, x, y, cv=cv_outer,scoring='neg_mean_absolute_error',n_jobs = 4)
            text_file.write('Mean and STD for MAE: %.3f (%.3f)\n' % (mean(MAE), std(MAE)))

            text_file.write ('------------------------------\n')
            text_file.write ('R2 values:\n')
            R2 = cross_val_score(search, x, y, cv=cv_outer,scoring='r2',n_jobs = 4)
            text_file.write('Mean and STD for R2: %.3f (%.3f)\n' % (mean(R2), std(R2)))


		    ############################
