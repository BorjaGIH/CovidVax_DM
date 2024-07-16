from importlib import reload
import numpy as np
import os
import pandas as pd
import pygformula
from pygformula import ParametricGformula
from pygformula.parametric_gformula.interventions import static
from pygformula.data import load_basicdata_nocomp
import pickle
import gc
import pyarrow as pa
import pyarrow.parquet as pq
import statsmodels.formula.api as smf
import statsmodels.api as sm
from datetime import datetime

# Time
startTime = datetime.now()

# Read and prepare data
data = pd.read_csv('included_cohort_prep.csv')
data.drop(['Unnamed: 0'], inplace=True, axis=1)
data = data.sample(frac=0.05, random_state=1).copy()
data.sexe = data.sexe.map({'H':0, 'D':1})

###### CAMBIAR FORMATO DE LOS DATOS (PIVOT) 
for i in range(1,4):
  print(i)
  data.rename({'VACUNA_{}_DATA'.format(i): 'VACUNA_DATA_{}'.format(i)}, axis=1, inplace=True)
  data.rename({'VACUNA_{}_MOTIU'.format(i): 'VACUNA_MOTIU_{}'.format(i)}, axis=1, inplace=True)
  data.rename({'VACUNA_{}_DATA_pp'.format(i): 'VACUNA_DATA_pp_{}'.format(i)}, axis=1, inplace=True)
  
stubnames = ['VACUNA_DATA_1', 'VACUNA_MOTIU_1', 'VACUNA_DATA_2',
'VACUNA_MOTIU_2', 'VACUNA_DATA_3', 'VACUNA_MOTIU_3', 'VACUNA_DATA_pp_1',
'VACUNA_DATA_pp_2', 'VACUNA_DATA_pp_3', 
'test_date_covid_1', 'test_res_covid_1',
'test_date_covid_2', 'test_res_covid_2', 'test_date_covid_3',
'test_res_covid_3', 'test_date_imc_1', 'test_res_imc_1',
'test_date_imc_2', 'test_res_imc_2', 'test_date_imc_3',
'test_res_imc_3', 'test_date_sp_1', 'test_res_sp_1', 'test_date_sp_2',
'test_res_sp_2', 'test_date_sp_3', 'test_res_sp_3', 'test_date_dp_1',
'test_res_dp_1', 'test_date_dp_2', 'test_res_dp_2', 'test_date_dp_3',
'test_res_dp_3', 'test_date_abdo_1', 'test_res_abdo_1',
'test_date_abdo_2', 'test_res_abdo_2', 'test_date_abdo_3',
'test_res_abdo_3', 'test_date_bg_1', 'test_res_bg_1', 'test_date_bg_2',
'test_res_bg_2', 'test_date_bg_3', 'test_res_bg_3', 'test_date_chol_1',
'test_res_chol_1', 'test_date_chol_2', 'test_res_chol_2',
'test_date_chol_3', 'test_res_chol_3', 'test_date_smoking_1',
'test_res_smoking_1', 'test_date_smoking_2', 'test_res_smoking_2',
'test_date_smoking_3', 'test_res_smoking_3', 'test_date_gma_1',
'test_res_gma_1', 'test_date_gma_2', 'test_res_gma_2',
'test_date_gma_3', 'test_res_gma_3', 'Vacuna_1', 'Vacuna_2', 'Vacuna_3']

for i in range(0,len(stubnames)):
  stubnames[i] = stubnames[i][0:-1]
  
data_piv = pd.wide_to_long(data, list(set(stubnames)), i='NIA', j='time')
data_piv.reset_index(inplace=True, drop=False)
data_piv.time = data_piv.time - 1

# Transform outcome for binary eof
data_piv.loc[(data_piv.time==1)|(data_piv.time==2), 'DM'] = np.NaN

# G-formula package things
time_name = 'time'
id = 'NIA'
time_points = np.max(np.unique(data_piv[time_name])) + 1

covnames = [
  'test_res_sp_',
  'test_res_smoking_',
  'test_res_chol_',
  'test_res_abdo_',
  'test_res_dp_',
  'test_res_imc_',
  'test_res_bg_',
  'test_res_covid_',
  'test_res_gma_',
  'Vacuna_']

covtypes = [
  'unknown-continuous',
  'unknown-continuous',
  'unknown-continuous',
  'unknown-continuous',
  'unknown-continuous',
  'unknown-continuous',
  'unknown-continuous',
  'unknown-binary',
  'unknown-continuous',
  'unknown-binary']

trunc_params = ['NA', 'NA', 'NA', 'NA', 'NA', 'NA','NA', 'NA', 'NA', 'NA']

covmodels = [
  'test_res_sp_ ~ lag1_test_res_sp_ + lag1_test_res_imc_ + data_naixement + pais_c + lag1_test_res_smoking_ + test_res_sociostat_1 + lag1_Vacuna_ + time',
  'test_res_smoking_ ~ lag1_test_res_smoking_ + data_naixement + pais_c + test_res_sociostat_1 + lag1_Vacuna_ + time',
  'test_res_chol_ ~ lag1_test_res_chol_ + data_naixement + pais_c + test_res_sociostat_1 + lag1_Vacuna_ + time',
  'test_res_abdo_ ~ lag1_test_res_abdo_ + lag1_test_res_imc_ + data_naixement + pais_c + lag1_test_res_smoking_ + lag1_Vacuna_ + time',
  'test_res_dp_ ~ lag1_test_res_dp_ + lag1_test_res_imc_ + data_naixement + pais_c + lag1_test_res_smoking_ + test_res_sociostat_1 + lag1_Vacuna_ + time',
  'test_res_imc_ ~ lag1_test_res_imc_ + data_naixement + pais_c + lag1_test_res_smoking_ + test_res_sociostat_1 + lag1_Vacuna_ + time',
  'test_res_bg_ ~ lag1_test_res_bg_ + lag1_test_res_imc_ + data_naixement + pais_c + test_res_sociostat_1 + lag1_Vacuna_ + time',
  'test_res_covid_ ~ lag1_test_res_covid_ + lag1_test_res_imc_ + data_naixement + pais_c + lag1_test_res_bg_ + lag1_test_res_smoking_ + test_res_sociostat_1 + lag1_Vacuna_ + time',
  'test_res_gma_ ~ lag1_test_res_gma_ + lag1_test_res_imc_ + data_naixement + lag1_test_res_dp_ + lag1_test_res_sp_ + lag1_test_res_chol_ + pais_c + lag1_test_res_bg_ + lag1_test_res_smoking_ + lag1_Vacuna_ + time',
  'Vacuna_ ~ lag1_Vacuna_ + lag1_test_res_sp_ + lag1_test_res_smoking_ + lag1_test_res_chol_ + lag1_test_res_abdo_ + lag1_test_res_dp_ + lag1_test_res_imc_ + lag1_test_res_bg_ + lag1_test_res_covid_ + lag1_test_res_gma_ + time']

basecovs = ['abs_c', 'pais_c', 'sexe', 'data_naixement', 'test_res_sociostat_1'] 

outcome_name = 'DM'
ymodel = 'DM ~ abs_c + pais_c + sexe + data_naixement + test_res_sociostat_1+\
  test_res_sp_ + lag1_test_res_sp_ +\
  test_res_smoking_ + lag1_test_res_smoking_+\
  test_res_chol_ + lag1_test_res_chol_+\
  test_res_abdo_ + lag1_test_res_abdo_+\
  test_res_dp_ + lag1_test_res_dp_+\
  test_res_imc_ + lag1_test_res_imc_+\
  test_res_bg_ + lag1_test_res_bg_+\
  test_res_covid_ + lag1_test_res_covid_+\
  test_res_gma_ + lag1_test_res_gma_+\
  Vacuna_ + lag1_Vacuna_'
 
outcome_type = 'binary_eof'
int_descript = ['Never treat', 'Treat on Vacuna only at t1', 'Treat on Vacuna only at t1 & t2', 'Treat on Vacuna at t1, t2 & t3']
ymodel_type = 'ML'

Intervention1_Vacuna_ = [static, np.zeros(time_points),[0, 1, 2]]
Intervention2_Vacuna_ = [static, np.ones(time_points), [0]]
Intervention3_Vacuna_ = [static, np.ones(time_points), [0, 1]]
Intervention4_Vacuna_ = [static, np.ones(time_points), [0, 1, 2]]


# Try to save some space - gives a Future warning of setting incompatible type
#float64_cols = list(data_piv.select_dtypes(include='float64'))
#data_piv[float64_cols] = data_piv[float64_cols].astype('float32')
gc.collect()

# G-formula package call
g = ParametricGformula(obs_data = data_piv, id = id, time_name=time_name,
             time_points = time_points, int_descript = int_descript,
             covnames=covnames, covtypes=covtypes, trunc_params=trunc_params,
             covmodels=covmodels, basecovs=basecovs,
             outcome_name=outcome_name, ymodel=ymodel, ymodel_type=ymodel_type, outcome_type=outcome_type,
             Intervention1_Vacuna_ = Intervention1_Vacuna_,
             Intervention2_Vacuna_ = Intervention2_Vacuna_,
             Intervention3_Vacuna_ = Intervention3_Vacuna_,
             Intervention4_Vacuna_ = Intervention4_Vacuna_,
             nsamples=10, parallel=True, ncores=30)
g.fit()

# Serialize the object to a binary format
with open('gformRF.pkl', 'wb') as file:
    pickle.dump(g, file)

# Print time
print(datetime.now() - startTime)
