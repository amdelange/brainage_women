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

###### ADD PATHS TO DATA AND FOLDERS WHERE FILES WILL BE SAVED
datapath=""

###### USE DICTIONARIES TO LOOP OVER ALL DIFFUSION MODELS AND RUN BRAIN AGE PREDICTION
dict = {}
corr = {}
pred = {}


#Output dictionary to store r and CI in json (to use for LaTeX table in step 8)
out_dict = {}

sex = ['both','female','male']

# DEFINE dwMRI MODELS
model = ["global"]


for s in sex:
    for m in model:
        #saves the results to text files
        file=savepath+'Output_%s_%s.txt'% (m,s)
        with open(file, 'w') as text_file:

            text_file.write("===============\n")
            text_file.write("MODEL = %s\n" % m)
            text_file.write("===============\n")

            dict['%s' % m] = pd.read_csv(datapath+'%s_%s.csv' % (m,s))


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
                refit=True)


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

            #Get the best esimators from the search, and use those hyper params in a final model (which we will then use for feature importance)
            best_est = search.best_estimator_
            print("BEST ESTIMATOR")
            print(best_est)

            #Use the final optimised values in a model
            final_model = XGBRegressor(learning_rate=0.05, max_depth=9, n_estimators=180, nthread=4, objective='reg:squarederror', seed=42)


            # RUN CROSS_VAL_PREDICT
            print ('Running cross_val_predict for %s %s' % (m,s))

            # DEFINE THE VARIABLES PRED (PREDICTED AGE) AND BAG (BRAIN AGE GAP)
            #pred[m] = cross_val_predict(search, x, y, cv=cv_outer, n_jobs=2)
            pred[m] = cross_val_predict(final_model, x, y, cv=cv_outer, n_jobs=2)

            BAG = pred[m] - y


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


            # CALCULATE BAG RESIDUALS
            print ('fitting data for age correction for %s' % m)
            z = np.polyfit(y, BAG, 1)
            resid = BAG - (z[1] + z[0]*y)
            x_copy["BAG_residual_%s" % m] = resid
            print (x_copy.head(5))


            # SAVE FILE
            print ('saving file with brain age estimates')
            x_copy_save.to_csv(datapath+'Predictions_%s_%s.csv' % (m,s), sep=',',index=None)

            #Store r and CI in dict for writing to json
            out_dict[m] = corr[m]


            #Write the JSON with r and CI values to json
            with open(savepath+'r_vals_tracts_dict_%s_%s.json' %(m,s),'w') as f:
                json.dump(out_dict, f, sort_keys=True, indent=4)



            #Get the feature importances

            text_file.write ('Feature importance for %s\n' % m)

            result = final_model.fit(x, y)

            print(result.best_estimator_)

            feature_importances = result.get_booster().get_score(importance_type='gain')

            keys = list(feature_importances.keys())
            values = list(feature_importances.values())

            data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
            text_file.write (data)
            print (data)
