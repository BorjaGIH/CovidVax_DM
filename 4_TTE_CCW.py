import numpy as np
import pandas as pd
import gc
import datetime
import joblib
import os
import shutil
from scipy.stats import normaltest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import glob
import tempfile
import statsmodels.api as sm
import datetime


# Start timer
startTime = datetime.datetime.now()

# Define weight computing function
def compute_CCWeights2(data, end_time, baseline_covariates, covariates):
    weights_num_t_model = np.ones(len(data))
    weights_den_t_model = np.ones(len(data))
    
    for i in range(1,end_time + 1):
        
        if i==1:
            regression_num = LogisticRegression(solver='newton-cholesky', max_iter=200).fit( data[baseline_covariates], data["Censor_{}".format(i)])
            weights_num_t_model = weights_num_t_model * regression_num.predict_proba(data[baseline_covariates])[:,np.where(regression_num.classes_==0)].flatten()

            regression_den = LogisticRegression(solver='newton-cholesky', max_iter=200).fit( data[baseline_covariates + ["{}{}".format(cov, (i)) for cov in covariates]], data["Censor_{}".format(i)])
            weights_den_t_model = weights_den_t_model * regression_den.predict_proba(data[baseline_covariates + ["{}{}".format(cov, (i)) for cov in covariates]])[:,np.where(regression_den.classes_==0)].flatten()
            
            
        else: # solver='newton-cholesky', max_iter=200
            regression_num = LogisticRegression(solver='newton-cholesky', max_iter=200).fit( data[data["Censor_{}".format(i-1)]==0][baseline_covariates] , data[data["Censor_{}".format(i-1)]==0]["Censor_{}".format(i)])
            #regression_num = RandomForestClassifier().fit( data[data["Censor_{}".format(i-1)]==0][baseline_covariates] , data[data["Censor_{}".format(i-1)]==0]["Censor_{}".format(i)])
            temp_weights_num = regression_num.predict_proba(data[data["Censor_{}".format(i-1)]==0][baseline_covariates])[:,np.where(regression_num.classes_==0)].flatten()
            temp_weights_dim = np.ones(len(weights_num_t_model))
            temp_weights_dim[data["Censor_{}".format(i-1)]==0] = temp_weights_num
            weights_num_t_model = weights_num_t_model * temp_weights_dim

            regression_den = LogisticRegression(solver='newton-cholesky', max_iter=200).fit( data[data["Censor_{}".format(i-1)]==0][baseline_covariates + ["{}{}".format(cov, (i)) for cov in covariates]] , data[data["Censor_{}".format(i-1)]==0]["Censor_{}".format(i)])
            #regression_den = RandomForestClassifier().fit( data[data["Censor_{}".format(i-1)]==0][baseline_covariates + ["{}{}".format(cov, (i)) for cov in covariates]] , data[data["Censor_{}".format(i-1)]==0]["Censor_{}".format(i)])
            temp_weights_den = regression_den.predict_proba(data[data["Censor_{}".format(i-1)]==0][baseline_covariates + ["{}{}".format(cov, (i)) for cov in covariates]])[:,np.where(regression_den.classes_==0)].flatten()
            temp_weights_dim = np.ones(len(weights_num_t_model))
            temp_weights_dim[data["Censor_{}".format(i-1)]==0] = temp_weights_den
            weights_den_t_model = weights_den_t_model * temp_weights_dim
    
    weights = weights_num_t_model / weights_den_t_model
    
    return weights


# Setup info for saving path purposes and others
analysis_method = 'tte_ccw'
missing_assumption = 'mar' # 'struct_miss'
data_fraction = 'complete_data' # 
grace_period_days = 180
grace_period = datetime.timedelta(days=grace_period_days)

# Read and prepare data
all_data = pd.read_csv('/mnt/dicoms/borja_files/CovidVax_DM/data/currentData11072025/included_cohort_prep_{}.csv'.format(missing_assumption))
all_data.drop(['Unnamed: 0'], inplace=True, axis=1)
if data_fraction=='sample_data':
  all_data = all_data.sample(frac=0.05, random_state=1)

# Preprocess
all_data['VACUNA_1_DATA'] = pd.to_datetime(all_data['VACUNA_1_DATA'])
all_data['VACUNA_2_DATA'] = pd.to_datetime(all_data['VACUNA_2_DATA'])
all_data['VACUNA_3_DATA'] = pd.to_datetime(all_data['VACUNA_3_DATA'])
all_data['VACUNA_1_DATA_pp'] = pd.to_datetime(all_data['VACUNA_1_DATA_pp'], format='mixed', utc=True)
all_data['VACUNA_2_DATA_pp'] = pd.to_datetime(all_data['VACUNA_2_DATA_pp'], format='mixed', utc=True)
all_data['VACUNA_3_DATA_pp'] = pd.to_datetime(all_data['VACUNA_3_DATA_pp'], format='mixed', utc=True)

# Clone
data0 = all_data.copy()
data1 = all_data.copy()
data2 = all_data.copy()
data3 = all_data.copy()

for index, clonetable in enumerate([data0, data1, data2, data3]):
  clonetable.NIA = clonetable.NIA.astype(str) + f'_{index}'

  clonetable['Vacuna_assign_1'] = 0
  clonetable['Vacuna_assign_2'] = 0
  clonetable['Vacuna_assign_3'] = 0
  clonetable['N_vaccine_total_assigned'] = index

  for j in range(1,index+1):
    clonetable[f'Vacuna_assign_{j}'] = 1

  clonetable['Censor_1'] = 0 
  clonetable['Censor_2'] = 0
  clonetable['Censor_3'] = 0

  # Censor violations of the protocol
  clonetable.loc[clonetable.Vacuna_1!=clonetable.Vacuna_assign_1, ['Censor_1', 'Censor_2', 'Censor_3']] = 1
  clonetable.loc[clonetable.Vacuna_2!=clonetable.Vacuna_assign_2, ['Censor_2', 'Censor_3']] = 1
  clonetable.loc[clonetable.Vacuna_3!=clonetable.Vacuna_assign_3, ['Censor_3']] = 1

  # Censor violations of the grace period
  clonetable.loc[clonetable.VACUNA_1_DATA>(clonetable.VACUNA_1_DATA_pp + grace_period), ['Censor_1', 'Censor_2', 'Censor_3']] = 1
  clonetable.loc[clonetable.VACUNA_2_DATA>(clonetable.VACUNA_2_DATA_pp + grace_period), ['Censor_2', 'Censor_3']] = 1
  clonetable.loc[clonetable.VACUNA_3_DATA>(clonetable.VACUNA_3_DATA_pp + grace_period), ['Censor_3']] = 1

data = pd.concat([data0, data1, data2, data3])

del data0, data1, data2, data3, clonetable, all_data
gc.collect()

# Weighting
end_time = 3
baseline_covariates = ['abs_c', 'pais_c', 'sexe', 'data_naixement', 'test_res_sociostat_1']
covariates = ['test_res_sp_', 'test_res_smoking_', 'test_res_chol_', 'test_res_abdo_', 'test_res_dp_', 'test_res_imc_', 'test_res_bg_', 'test_res_covid_', 'test_res_gma_']

# Call function for getting the weights
data['weights'] = compute_CCWeights2(data, end_time=end_time, baseline_covariates=baseline_covariates, covariates=covariates)

# Drop infs
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(subset=["weights"], how="all", inplace=True)
data.reset_index(inplace=True, drop=False)

# Outcome model
data['DM_1']=pd.Series(np.zeros(len(data)))
data['DM_2']=pd.Series(np.zeros(len(data)))
data['DM_3']=pd.Series(np.zeros(len(data)))

data.loc[((data.DM==1) & (data.N_vaccine_total==1) & (((data.DATA_DM_min<data.VACUNA_2_DATA)) & ((data.DATA_DM_min>data.VACUNA_1_DATA)))), 'DM_1'] = 1
data.loc[((data.DM==1) & (data.N_vaccine_total==2) & (((data.DATA_DM_min<data.VACUNA_3_DATA)) & ((data.DATA_DM_min>data.VACUNA_2_DATA)))), 'DM_2'] = 1
data.loc[((data.DM==1) & (data.N_vaccine_total==3) & ((data.DATA_DM_min>data.VACUNA_3_DATA))), 'DM_3'] = 1

# Weighted pooled logistic regression
# 1. Pivot the data for this
# 2. Regress DM on the explanatory vars (including time, pooling var), including the previously computed weights in the regression

###### PIVOT DATA
for i in range(1,4):
    data.rename({'VACUNA_{}_DATA'.format(i): 'VACUNA_DATA_{}'.format(i)}, axis=1, inplace=True)
    data.rename({'VACUNA_{}_MOTIU'.format(i): 'VACUNA_MOTIU_{}'.format(i)}, axis=1, inplace=True)
    data.rename({'VACUNA_{}_DATA_pp'.format(i): 'VACUNA_DATA_pp_{}'.format(i)}, axis=1, inplace=True)

stubnames = ['test_date_covid_1', 'test_res_covid_1','test_date_covid_2', 'test_res_covid_2', 'test_date_covid_3', 'test_res_covid_3', 
             'test_date_imc_1', 'test_res_imc_1','test_date_imc_2', 'test_res_imc_2', 'test_date_imc_3', 'test_res_imc_3', 
             'test_date_sp_1', 'test_res_sp_1', 'test_date_sp_2','test_res_sp_2', 'test_date_sp_3', 'test_res_sp_3', 
             'test_date_dp_1','test_res_dp_1', 'test_date_dp_2', 'test_res_dp_2', 'test_date_dp_3', 'test_res_dp_3', 
             'test_date_abdo_1', 'test_res_abdo_1','test_date_abdo_2', 'test_res_abdo_2', 'test_date_abdo_3', 'test_res_abdo_3', 
             'test_date_bg_1', 'test_res_bg_1', 'test_date_bg_2','test_res_bg_2', 'test_date_bg_3', 'test_res_bg_3', 
             'test_date_chol_1', 'test_res_chol_1', 'test_date_chol_2', 'test_res_chol_2', 'test_date_chol_3', 'test_res_chol_3', 
             'test_date_smoking_1', 'test_res_smoking_1', 'test_date_smoking_2', 'test_res_smoking_2', 'test_date_smoking_3', 'test_res_smoking_3', 
             'test_date_gma_1', 'test_res_gma_1', 'test_date_gma_2', 'test_res_gma_2', 'test_date_gma_3', 'test_res_gma_3',  
             'Vacuna_1', 'Vacuna_2', 'Vacuna_3', 
             'Censor_1', 'Censor_2', 'Censor_3', 
             'DM_1', 'DM_2', 'DM_3']

for i in range(0,len(stubnames)):
    stubnames[i] = stubnames[i][0:-1]

# For the outcome model, use only the uncensored at C_3
data = data[data['Censor_3']==0]

# Pivot
data_piv = pd.wide_to_long(data, list(set(stubnames)), i='NIA', j='time')
data_piv.reset_index(inplace=True, drop=False)

del data
gc.collect()

# Weighted regression
outcome_model_tv_vars = [
    'test_res_sp_',
    'test_res_smoking_',
    'test_res_chol_',
    'test_res_abdo_',
    'test_res_dp_',
    'test_res_imc_',
    'test_res_bg_',
    'test_res_covid_',
    'test_res_gma_',
    'time',
    'Vacuna_',]

outcome_model_base_vars = ['abs_c', 'pais_c', 'sexe', 'data_naixement', 'test_res_sociostat_1']

# Fit outcome model
outcome_model = RandomForestClassifier(n_jobs=20).fit(data_piv[outcome_model_vars + baseline_vars], data_piv['DM_'], data_piv['weights'])

# Compute DM risk (prob) under different interventions

# Print time
print(datetime.datetime.now() - startTime)