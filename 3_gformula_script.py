import numpy as np
import pandas as pd
import gc
import datetime
import joblib
from pygformula import ParametricGformula
from pygformula.parametric_gformula.interventions import static
from sklearn.preprocessing import StandardScaler
import os
import shutil
import glob

# Start timer
startTime = datetime.datetime.now()

# Setup info for saving path purposes and others
analysis_method = 'gformula'
missing_assumption = 'mar' # 'struct_miss' or 'mar'
data_fraction = 'complete' # 'complete' or 'sample'
modelling = 'ML' # 'GLM' or 'ML'
save_results = True
normalize_data = False
parallel = True

# Print some params
print('Model type: {}'.format(modelling))
print('Normalize data: {}'.format(normalize_data))
print('Data: {}'.format(data_fraction))

# Loop over files
for file in glob.glob("/mnt/dicoms/borja_files/CovidVax_DM/data/currentData02092025/chunks_100_{}/*.csv".format(missing_assumption)):
#for i in range(76,101):
  #file = "/mnt/dicoms/borja_files/CovidVax_DM/data/currentData11072025/chunks_100_strucmiss/included_cohort_prep_struc_miss_{}.csv".format(i)
    
  # Read data
  data = pd.read_csv(file)
  
  # If sampling
  if data_fraction == 'sample':
    data = data.sample(frac=0.1)
    data.reset_index(inplace=True, drop=True)
  
  # Number of Monte-Carlo simulations
  #n_simul = int(np.ceil(0.1*len(data)))
  n_simul = len(data)
  
  ###### PIVOT DATA
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
  'Vacuna_assign_1', 'Vacuna_assign_2', 'Vacuna_assign_3'] 

  for i in range(0,len(stubnames)):
    stubnames[i] = stubnames[i][0:-1]

  data_piv = pd.wide_to_long(data, list(set(stubnames)), i='NIA', j='time')
  data_piv.rename({'Vacuna_':'Vacuna'}, axis=1, inplace=True)
  data_piv.reset_index(inplace=True, drop=False)
  data_piv.time = data_piv.time - 1
    
  del data
  gc.collect()

  # G-formula params
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

  if missing_assumption == 'mar':
    
    if modelling == 'ML':
      covtypes = [
      'unknown-continuous',
      'unknown-categorical',
      'unknown-continuous',
      'unknown-continuous',
      'unknown-continuous',
      'unknown-continuous',
      'unknown-continuous',
      'unknown-binary',
      'unknown-continuous',
      'unknown-binary']
      ymodel_type = 'ML'
    elif modelling == 'GLM':
      covtypes = [
      'normal',
      'categorical',
      'normal',
      'normal',
      'normal',
      'normal',
      'normal',
      'binary',
      'normal',
      'binary']
      ymodel_type = 'GLM'
    else:
      raise('Wrong modelling option, choose between "ML" or "GLM"')
    
    covmodels = [
      'test_res_sp_ ~ lag1_test_res_sp_ + lag1_test_res_imc_ + data_naixement + C(pais_c) + C(lag1_test_res_smoking_) + C(test_res_sociostat_1) + lag1_Vacuna + time',
      'test_res_smoking_ ~ C(lag1_test_res_smoking_) + data_naixement + C(pais_c) + C(test_res_sociostat_1) + lag1_Vacuna + time',
      'test_res_chol_ ~ lag1_test_res_chol_ + data_naixement + C(pais_c) + C(test_res_sociostat_1) + lag1_Vacuna + time',
      'test_res_abdo_ ~ lag1_test_res_abdo_ + lag1_test_res_imc_ + data_naixement + C(pais_c) + C(lag1_test_res_smoking_) + lag1_Vacuna + time',
      'test_res_dp_ ~ lag1_test_res_dp_ + lag1_test_res_imc_ + data_naixement + C(pais_c) + C(lag1_test_res_smoking_) + C(test_res_sociostat_1) + lag1_Vacuna + time',
      'test_res_imc_ ~ lag1_test_res_imc_ + data_naixement + C(pais_c) + C(lag1_test_res_smoking_) + C(test_res_sociostat_1) + lag1_Vacuna + time',
      'test_res_bg_ ~ lag1_test_res_bg_ + lag1_test_res_imc_ + data_naixement + C(pais_c) + C(test_res_sociostat_1) + lag1_Vacuna + time',
      'test_res_covid_ ~ lag1_test_res_covid_ + lag1_test_res_imc_ + data_naixement + C(pais_c) + lag1_test_res_bg_ + C(lag1_test_res_smoking_) + C(test_res_sociostat_1) + lag1_Vacuna + time',
      'test_res_gma_ ~ lag1_test_res_gma_ + lag1_test_res_imc_ + data_naixement + lag1_test_res_dp_ + lag1_test_res_sp_ + lag1_test_res_chol_ + C(pais_c) + lag1_test_res_bg_ + C(lag1_test_res_smoking_) + lag1_Vacuna + time',
      #'Vacuna ~ lag1_Vacuna + lag1_test_res_sp_ + C(lag1_test_res_smoking_) + lag1_test_res_chol_ + lag1_test_res_abdo_ + lag1_test_res_dp_ + lag1_test_res_imc_ + lag1_test_res_bg_ + lag1_test_res_covid_ + lag1_test_res_gma_ + time'
      'Vacuna ~ lag1_Vacuna + test_res_sp_ + C(test_res_smoking_) + test_res_chol_ + test_res_abdo_ + test_res_dp_ + test_res_imc_ + test_res_bg_ + test_res_covid_ + test_res_gma_ \
                + C(abs_c) + C(pais_c) + sexe + data_naixement + C(test_res_sociostat_1) + time']

    ymodel = 'DM ~ C(abs_c) + C(pais_c) + sexe + data_naixement + C(test_res_sociostat_1)+\
      test_res_sp_ +\
      C(test_res_smoking_) +\
      test_res_chol_ +\
      test_res_abdo_ +\
      test_res_dp_ +\
      test_res_imc_ +\
      test_res_bg_ +\
      test_res_covid_ +\
      test_res_gma_ +\
      Vacuna + lag1_Vacuna + lag2_Vacuna + time'
    
  elif missing_assumption == 'struct_miss':
    
    if modelling == 'ML':
      covtypes = [
      'unknown-continuous',
      'unknown-categorical',
      'unknown-continuous',
      'unknown-categorical',
      'unknown-continuous',
      'unknown-continuous',
      'unknown-continuous',
      'unknown-binary',
      'unknown-continuous',
      'unknown-binary']
      ymodel_type = 'ML'
    elif modelling == 'GLM':
      covtypes = [
      'normal',
      'categorical',
      'normal',
      'categorical',
      'normal',
      'normal',
      'normal',
      'binary',
      'normal',
      'binary']
      ymodel_type = 'GLM'
    else:
      raise('Wrong modelling option, choose between "ML" or "GLM"')
    
    covmodels = [
      'test_res_sp_ ~ lag1_test_res_sp_ + lag1_test_res_imc_ + data_naixement + C(pais_c) + C(lag1_test_res_smoking_) + C(test_res_sociostat_1) + lag1_Vacuna + time',
      'test_res_smoking_ ~ C(lag1_test_res_smoking_) + data_naixement + C(pais_c) + C(test_res_sociostat_1) + lag1_Vacuna + time',
      'test_res_chol_ ~ lag1_test_res_chol_ + data_naixement + C(pais_c) + C(test_res_sociostat_1) + lag1_Vacuna + time',
      'test_res_abdo_ ~ C(lag1_test_res_abdo_) + lag1_test_res_imc_ + data_naixement + C(pais_c) + C(lag1_test_res_smoking_) + lag1_Vacuna + time',
      'test_res_dp_ ~ lag1_test_res_dp_ + lag1_test_res_imc_ + data_naixement + C(pais_c) + C(lag1_test_res_smoking_) + C(test_res_sociostat_1) + lag1_Vacuna + time',
      'test_res_imc_ ~ lag1_test_res_imc_ + data_naixement + C(pais_c) + C(lag1_test_res_smoking_) + C(test_res_sociostat_1) + lag1_Vacuna + time',
      'test_res_bg_ ~ lag1_test_res_bg_ + lag1_test_res_imc_ + data_naixement + C(pais_c) + C(test_res_sociostat_1) + lag1_Vacuna + time',
      'test_res_covid_ ~ lag1_test_res_covid_ + lag1_test_res_imc_ + data_naixement + C(pais_c) + lag1_test_res_bg_ + C(lag1_test_res_smoking_) + C(test_res_sociostat_1) + lag1_Vacuna + time',
      'test_res_gma_ ~ lag1_test_res_gma_ + lag1_test_res_imc_ + data_naixement + lag1_test_res_dp_ + lag1_test_res_sp_ + lag1_test_res_chol_ + C(pais_c) + lag1_test_res_bg_ + C(lag1_test_res_smoking_) + lag1_Vacuna + time',
      #'Vacuna ~ lag1_Vacuna + lag1_test_res_sp_ + C(lag1_test_res_smoking_) + lag1_test_res_chol_ + C(lag1_test_res_abdo_) + lag1_test_res_dp_ + lag1_test_res_imc_ + lag1_test_res_bg_ + lag1_test_res_covid_ + lag1_test_res_gma_ + time'
      'Vacuna ~ lag1_Vacuna + test_res_sp_ + C(test_res_smoking_) + test_res_chol_ + C(test_res_abdo_) + test_res_dp_ + test_res_imc_ + test_res_bg_ + test_res_covid_ + test_res_gma_ \
                + C(abs_c) + C(pais_c) + sexe + data_naixement + C(test_res_sociostat_1) + time']

    ymodel = 'DM ~ C(abs_c) + C(pais_c) + sexe + data_naixement + C(test_res_sociostat_1)+\
      test_res_sp_ +\
      C(test_res_smoking_) +\
      test_res_chol_ +\
      C(test_res_abdo_) +\
      test_res_dp_ +\
      test_res_imc_ +\
      test_res_bg_ +\
      test_res_covid_ +\
      test_res_gma_ +\
      Vacuna + lag1_Vacuna + lag2_Vacuna + time'
  else:
    raise('Wrong missing assumption option, choose between "mar" or "struct_miss"')

  trunc_params = ['NA', 'NA', 'NA', 'NA', 'NA', 'NA','NA', 'NA', 'NA', 'NA']
  basecovs = ['abs_c', 'pais_c', 'sexe', 'data_naixement', 'test_res_sociostat_1']
  outcome_name = 'DM'
  
  outcome_type = 'binary_eof'
  if outcome_type=='binary_eof':
    # Transform outcome for binary eof
    data_piv.loc[(data_piv.time==0)|(data_piv.time==1), 'DM'] = np.NaN

  # Interventions
  int_descript = ['Never treat', 'Treat on Vacuna only at t1', 'Treat on Vacuna only at t1 & t2', 'Treat on Vacuna at t1, t2 & t3']
  
  Intervention1_Vacuna = [static, (0, 0, 0), [0, 1, 2]]
  Intervention2_Vacuna = [static, (1, 0, 0), [0, 1, 2]]
  Intervention3_Vacuna = [static, (1, 1, 0), [0, 1, 2]]
  Intervention4_Vacuna = [static, (1, 1, 1), [0, 1, 2]]
  
  # Normalize data
  if normalize_data:
    data_piv[[covnames[covtypes=='normal']] + ['data_naixement', 'test_res_sociostat_1']] = StandardScaler().fit_transform(data_piv[[covnames[covtypes=='normal']] + ['data_naixement', 'test_res_sociostat_1']])

  # G-formula package call
  g = ParametricGformula(obs_data=data_piv, id=id, time_name=time_name, time_points=time_points, 
                         covnames=covnames, covtypes=covtypes, covmodels=covmodels, basecovs=basecovs,
                         outcome_name=outcome_name, outcome_type=outcome_type, ymodel=ymodel, ymodel_type=ymodel_type, 
                         int_descript=int_descript, 
                         Intervention1_Vacuna = Intervention1_Vacuna,
                         Intervention2_Vacuna = Intervention2_Vacuna,
                         Intervention3_Vacuna = Intervention3_Vacuna,
                         Intervention4_Vacuna = Intervention4_Vacuna,
                         trunc_params=trunc_params, nsamples=0, parallel=parallel, parallel_bootstrap=False, ncores=32, n_simul=n_simul, save_results=save_results, 
                         save_path=f'Results/{data_fraction}_data/{analysis_method}/{missing_assumption}/out_modeltype_{modelling}/presplit_chunks_100')  
  
  g.fit()  

  #joblib.dump(g, f'pygformula_object_{c}.pickle')
  
  # Remove temporary files etc.
  for folder in ['/home/bvelasco/CovidVax_DM/temp/', '/home/bvelasco/.local/share/Trash/files']:
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
  
  del data_piv
  gc.collect()

# Print time
print(datetime.datetime.now() - startTime)