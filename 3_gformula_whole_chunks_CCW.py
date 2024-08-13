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
import datetime

# Time
startTime = datetime.datetime.now()

# Read and prepare data
all_data = pd.read_csv('included_cohort_prep.csv')
all_data.drop(['Unnamed: 0'], inplace=True, axis=1)

grace_period_days = 360
grace_period = datetime.timedelta(days=grace_period_days)

all_data['VACUNA_1_DATA'] = pd.to_datetime(all_data['VACUNA_1_DATA'])
all_data['VACUNA_2_DATA'] = pd.to_datetime(all_data['VACUNA_2_DATA'])
all_data['VACUNA_3_DATA'] = pd.to_datetime(all_data['VACUNA_3_DATA'])
all_data['VACUNA_1_DATA_pp'] = pd.to_datetime(all_data['VACUNA_1_DATA_pp'], format='mixed', utc=True)
all_data['VACUNA_2_DATA_pp'] = pd.to_datetime(all_data['VACUNA_2_DATA_pp'], format='mixed', utc=True)
all_data['VACUNA_3_DATA_pp'] = pd.to_datetime(all_data['VACUNA_3_DATA_pp'], format='mixed', utc=True)

# Process data by chunks
n_chunks = 10
remaining_index = pd.Series(all_data.index.values)
chunk_length = np.floor(len(remaining_index)/n_chunks)

for c in range(9,n_chunks):
  chunk = remaining_index.sample(n=int(chunk_length))
  remaining_index = remaining_index.drop(chunk)
  chunk_data = all_data.iloc[chunk].copy()
    
  # Clone
  data0 = chunk_data.copy()
  data1 = chunk_data.copy()
  data2 = chunk_data.copy()
  data3 = chunk_data.copy()
    
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

  print('% of uncensored')
  print(len(data[data.Censor_3==0]) / (len(data)/4) *100)

  data.data_naixement = data.data_naixement.str[0:4]
    
  ###### PIVOT 
  for i in range(1,4):
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
'test_date_gma_3', 'test_res_gma_3',  'Vacuna_1', 'Vacuna_2', 'Vacuna_3', 
'Vacuna_assign_1', 'Vacuna_assign_2', 'Vacuna_assign_3', 'Censor_1', 'Censor_2', 'Censor_3'] 

  for i in range(0,len(stubnames)):
    stubnames[i] = stubnames[i][0:-1]

  data_piv = pd.wide_to_long(data, list(set(stubnames)), i='NIA', j='time')
  data_piv.rename({'Vacuna_':'Vacuna'}, axis=1, inplace=True)
  data_piv.reset_index(inplace=True, drop=False)
  data_piv.time = data_piv.time - 1

  # G-formula package params
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
    'Vacuna']

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
    'test_res_sp_ ~ lag1_test_res_sp_ + lag1_test_res_imc_ + data_naixement + pais_c + lag1_test_res_smoking_ + test_res_sociostat_1 + lag1_Vacuna + time',
    'test_res_smoking_ ~ lag1_test_res_smoking_ + data_naixement + pais_c + test_res_sociostat_1 + lag1_Vacuna + time',
    'test_res_chol_ ~ lag1_test_res_chol_ + data_naixement + pais_c + test_res_sociostat_1 + lag1_Vacuna + time',
    'test_res_abdo_ ~ lag1_test_res_abdo_ + lag1_test_res_imc_ + data_naixement + pais_c + lag1_test_res_smoking_ + lag1_Vacuna + time',
    'test_res_dp_ ~ lag1_test_res_dp_ + lag1_test_res_imc_ + data_naixement + pais_c + lag1_test_res_smoking_ + test_res_sociostat_1 + lag1_Vacuna + time',
    'test_res_imc_ ~ lag1_test_res_imc_ + data_naixement + pais_c + lag1_test_res_smoking_ + test_res_sociostat_1 + lag1_Vacuna + time',
    'test_res_bg_ ~ lag1_test_res_bg_ + lag1_test_res_imc_ + data_naixement + pais_c + test_res_sociostat_1 + lag1_Vacuna + time',
    'test_res_covid_ ~ lag1_test_res_covid_ + lag1_test_res_imc_ + data_naixement + pais_c + lag1_test_res_bg_ + lag1_test_res_smoking_ + test_res_sociostat_1 + lag1_Vacuna + time',
    'test_res_gma_ ~ lag1_test_res_gma_ + lag1_test_res_imc_ + data_naixement + lag1_test_res_dp_ + lag1_test_res_sp_ + lag1_test_res_chol_ + pais_c + lag1_test_res_bg_ + lag1_test_res_smoking_ + lag1_Vacuna + time',
    'Vacuna ~ lag1_Vacuna + lag1_test_res_sp_ + lag1_test_res_smoking_ + lag1_test_res_chol_ + lag1_test_res_abdo_ + lag1_test_res_dp_ + lag1_test_res_imc_ + lag1_test_res_bg_ + lag1_test_res_covid_ + lag1_test_res_gma_ + time']

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
    Vacuna + lag1_Vacuna'
  ymodel_type = 'ML'


  outcome_type = 'binary_eof'
  if outcome_type=='binary_eof':
    # Transform outcome for binary eof
    data_piv.loc[(data_piv.time==0)|(data_piv.time==1), 'DM'] = np.NaN

  censor_CCW_name = 'Censor_'
  censor_CCW_model = 'Censor_ ~  test_res_sp_ + test_res_smoking_ + test_res_chol_ + test_res_abdo_ + test_res_dp_ + test_res_imc_ + test_res_bg_ + test_res_covid_ + test_res_gma_+\
                    abs_c + pais_c + sexe + data_naixement + test_res_sociostat_1 + time ' # + lag1_Censor_


  int_descript = ['Never treat', 'Treat on Vacuna only at t1', 'Treat on Vacuna only at t1 & t2', 'Treat on Vacuna at t1, t2 & t3']
  Intervention1_Vacuna = [static, np.zeros(time_points),[0, 1, 2]]
  Intervention2_Vacuna = [static, np.ones(time_points), [0]]
  Intervention3_Vacuna = [static, np.ones(time_points), [0, 1]]
  Intervention4_Vacuna = [static, np.ones(time_points), [0, 1, 2]]


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
               censor_CCW_name = censor_CCW_name, censor_CCW_model = censor_CCW_model,
               Intervention1_Vacuna = Intervention1_Vacuna,
               Intervention2_Vacuna = Intervention2_Vacuna,
               Intervention3_Vacuna = Intervention3_Vacuna,
               Intervention4_Vacuna = Intervention4_Vacuna,
               nsamples=0, parallel=True, ncores=32, save_results=True, save_path=f'Results/{outcome_type}/grace_period_{grace_period_days}/')
  g.fit()

  # Print time
  print(datetime.datetime.now() - startTime)
