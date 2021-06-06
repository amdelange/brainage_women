import csv
import pandas as pd
from functools import reduce
import numpy as np
import pingouin as pg
from pingouin import partial_corr
import scipy
from scipy.stats.stats import pearsonr
import time
from numpy import mean
from numpy import std
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.lines as mlines
from scipy import stats
import joblib
import sys, os
import statsmodels.api as sm
import json

def pearsonr_ci(x,y,alpha=0.05):
    r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi

import seaborn as sns
sns.set(color_codes=True)
sns.set(font_scale=2)
sns.set_style("white")

#from matplotlib import rc
#rc('font',**{'family':'serif','serif':['Roman']})
#rc('text', usetex=True)


###### PATHS (MODIFY TO ADD YOUR OWN PATHS)
datapath="/.../data/"
savepath="/.../results/"


###### DICTIONARIES
dict = {}
corr = {}
pred = {}


#Output dictionary to store r and CI in json (to use for LaTeX table)
out_dict = {}

# DEFINE MODELS
model = ['diffusion','T1']


for m in model:
    #saves the results to text files
    file=savepath+'Output_model_%s.txt'% m
    with open(file, 'w') as text_file:

        text_file.write("===============\n")
        text_file.write("MODEL = %s\n" % m)
        text_file.write("===============\n")

        dict['%s' % m] = pd.read_csv(datapath+'corrected_%s_data.csv' % m)


        # SPLIT THE FILE INTO X AND Y, WHERE X IS ALL THE MRI DATA AND Y IS AGE
        x = dict['%s' % m]
        print ('splitting data into x and y for %s' % m)
        y = x['Age']


        # MAKE A COPY OF THE DATA FRAME TO MERGE WITH ESTIMATED BRAIN AGE AT THE END OF SCRIPT
        x_copy = x.copy()


        # REMOVE VARIABLES FROM X THAT SHOULD NOT BE INCLUDED IN THE REGRESSOR
        x = x.drop('eid',1)
        x = x.drop('Age',1)


        # CHECK THAT X INCLUDES ONLY MRI VARIABLES, AND Y INCLUDES ONLY AGE
        print ('printing final x for %s' % m)
        print (x.head(5))
        print ('printing final y for %s' % m)
        print (y.head(5))


        ############################

        # RUN PCA TO REDUCE COMPUTATIONAL TIME FOR THE FULL MODEL

        print ('Running PCA')
        scaler = StandardScaler()
        scaler.fit(x)

        x = pd.DataFrame(scaler.transform(x), columns=x.columns, index=x.index)

        pca = PCA()
        pca.fit(x)

        n_comps = pca.n_components_
        print ("Number of PCA components : %s" % n_comps)

        columns = ['pca_%i' % i for i in range(n_comps)]
        x = pd.DataFrame(pca.transform(x), columns=columns, index=x.index)

        eigenvalues = pca.explained_variance_
        eigenvalues_list = eigenvalues.tolist()
        eigenvalues_top10 = eigenvalues[:10]
        print ("printing eigenvalues top ten")
        print (eigenvalues_top10)

        #count the number with eigenvalue above 1
        n_components = sum(i >= 1 for i in eigenvalues_list)
        print ("printing n components with eigenvalue above 1")
        print (n_components)

        explained_var = pca.explained_variance_ratio_
        explained_var_top10 = explained_var[:10]
        print ("printing explained variance top10 components")
        print (explained_var_top10)

        #summarise the list of values
        sum_tot = sum(explained_var)
        print ("printing sum of the list of values")
        print (sum_tot)

        #create a series from the list
        #explained_var_top10 = pd.Series.from_array(explained_var_top10)

        #cumulative variance
        sum_explained_var = np.cumsum(explained_var)



        if(m=="diffusion"):
            #explained variance of the top x components
            explained_var_200 = 100*sum_explained_var[199]
            print("Explained variance of the top 200: %s" % explained_var_200)

            print ('selecting top 200 PCA components')
            x = x.iloc[:, : 200]
            print (x.head(5))

        if(m=="T1"):
            #explained variance of the top x components
            explained_var_700 = 100*sum_explained_var[699]
            print("Explained variance of the top 700: %s" % explained_var_700)

            print ('selecting top 700 PCA components')
            x = x.iloc[:, : 700]
            print (x.head(5))



        ############################

        # SPECIFY MODEL

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

        # RUN CROSS_VAL_PREDICT
        print ('Running cross_val_predict for %s' % m)


        # DEFINE THE VARIABLES PRED (PREDICTED AGE) AND BAG (BRAIN AGE GAP)
        pred[m] = cross_val_predict(search, x, y, cv=cv_outer, n_jobs=2)
        BAG = pred[m] - y


        #Train model
        search.fit(x,y)

        #Save the model to a file
        #First delete old file if it exists
        if os.path.exists(savepath+'model_%s.joblib' % m):
            os.remove(savepath+'model_%s.joblib' % m)
        joblib.dump(search,savepath+'model_%s.joblib' % m)


        # ADD PREDICTED BRAIN AGE AND BRAIN AGE GAP TO X_COPY TO GET A FULL DATAFRAME WITH ALL VARIABLES
        x_copy['pred_age_%s' % m] = pred[m]
        x_copy['brainage_gap_%s' % m] = x_copy['pred_age_%s' % m] - y


        # RUN CORRELATION FOR PREDICTED VERSUS TRUE AGE
        text_file.write("===============\n")
        text_file.write ('results cross val predict %s model \n' % m)
        text_file.write("===============\n")
        text_file.write ('pearsons corr for predicted versus true age for %s\n' % m)

        corr[m] = pearsonr_ci(x_copy['Age'],x_copy['pred_age_%s' %m])
        text_file.write ("r = %.3f\n" % corr[m][0])
        text_file.write ("r p-value = %.3e\n" % corr[m][1])
        text_file.write ("r CI = [%.3f,%.3f]\n" % (corr[m][2],corr[m][3]))

        text_file.write("===============\n")

        # CALCULATE BAG RESIDUALS (THE BAG VALUES RESIDUALISED FOR CHRONOLOGICAL AGE)
        print ('fitting data for age correction for %s' % m)
        z = np.polyfit(y, BAG, 1)
        resid = BAG - (z[1] + z[0]*y)
        x_copy["BAG_residual_%s" % m] = resid
        print (x_copy.head(5))


        # SAVE FILE
        # First create a dataframe including only relevant variables
        x_copy_save = x_copy[['eid','Age','pred_age_%s' % m,'brainage_gap_%s' % m,'BAG_residual_%s' % m]]
        #print (x_copy_save.head(5))	#uncomment to check file content
        print ('saving file with brain age estimates')
        x_copy_save.to_csv(savepath +'Brainage_%s.csv' % m, sep=',',index=None)

        #Store r and CI in dict for writing to json
        out_dict[m] = corr[m]


####################################

#Write the JSON with r and CI values to json
with open(savepath+'r_vals_full_model_dict.json','w') as f:
	json.dump(out_dict, f, sort_keys=True, indent=4)
