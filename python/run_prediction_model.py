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

def pearsonr_ci(x,y,alpha=0.05):

    r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi



####################################

###### ADD PATHS TO DATA
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
        file=datapath+'Output_%s_%s_testset.txt'% (m,s)
        with open(file, 'w') as text_file:

            text_file.write("===============\n")
            text_file.write("MODEL = %s\n" % m)
            text_file.write("===============\n")

            dict['%s' % m] = pd.read_csv(datapath+'%s_%s_testset.csv' % (m,s))

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


            # SPECIFY MODEL

            cv = KFold(n_splits=10, shuffle=True, random_state=1)

            # define the model using parameters based on held-out validation sample
            model = xgb.XGBRegressor(objective= 'reg:squarederror',nthread=4,seed=42,learning_rate =0.05, max_depth=5, n_estimators=180,random_state=42)

            text_file.write ('validating %s model\n' % m)

            text_file.write ('------------------------------\n')
            text_file.write ('RMSE values:\n')
            RMSE = cross_val_score(model, x, y, cv=cv,scoring='neg_root_mean_squared_error',n_jobs = 4)
            text_file.write('Mean and STD for RMSE: %.3f (%.3f)\n' % (mean(RMSE), std(RMSE)))

            text_file.write ('------------------------------\n')
            text_file.write ('MAE values:\n')
            MAE = cross_val_score(model, x, y, cv=cv,scoring='neg_mean_absolute_error',n_jobs = 4)
            text_file.write('Mean and STD for MAE: %.3f (%.3f)\n' % (mean(MAE), std(MAE)))

            text_file.write ('------------------------------\n')
            text_file.write ('R2 values:\n')
            R2 = cross_val_score(model, x, y, cv=cv,scoring='r2',n_jobs = 4)
            text_file.write('Mean and STD for R2: %.3f (%.3f)\n' % (mean(R2), std(R2)))


		    ############################

            # RUN CROSS_VAL_PREDICT TO GET PREDICTIONS FOR ALL SUBJECTS
            print ('Running cross_val_predict for %s %s' % (m,s))

            # DEFINE THE VARIABLES PRED (PREDICTED AGE) AND BAG (BRAIN AGE GAP)
            pred[m] = cross_val_predict(model, x, y, cv=cv, n_jobs=2)

            # ADD PREDICTED BRAIN AGE AND BRAIN AGE GAP TO X_COPY TO GET A FULL DATAFRAME WITH ALL VARIABLES
            x_copy['pred_age_%s' % m] = pred[m]
            x_copy['brainage_gap_%s' % m] = x_copy['pred_age_%s' % m] - y


            # RUN CORRELATION FOR PREDICTED VERSUS TRUE AGE
            text_file.write ('running pearsons corr for predicted versus true age for %s\n' % m)

            #corr[m] = pg.corr(x_copy['Age'],x_copy['pred_age_%s' %m])
            corr[m] = pearsonr_ci(x_copy['Age'],x_copy['pred_age_%s' %m])
            text_file.write ("r = %s\n" % corr[m][0])
            text_file.write ("r p-value = %s\n" % corr[m][1])
            text_file.write ("r CI = [%s,%s]\n" % (corr[m][2],corr[m][3]))

            text_file.write ('------------------------------\n')


            # SAVE FILE
            x_copy.to_csv(datapath+'Predictions_%s_%s_testset.csv' % (m,s), sep=',',index=None)


            #Get feature importances
            text_file.write ('Feature importance for %s\n' % m)

            result = model.fit(x, y)

            feature_importances = result.get_booster().get_score(importance_type='gain')
            keys = list(feature_importances.keys())
            values = list(feature_importances.values())

            data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
            text_file.write (data)
            print (data)
