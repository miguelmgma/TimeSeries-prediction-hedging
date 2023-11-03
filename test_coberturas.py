import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

from src import preprocessing_functions as preproc
from src import config as conf
from src import aux_functions as aux
from src.plot_functions import grafico_columnas_agrupadas
from src.hedging import rw_hedging_forwards,rw_hedging_forwards_eval


### Data ingestion and preprocessing

df_c = preproc.df_preparation()

### User inputs

x_cols = [
        'CFI2c5', 
        'POOL AVG',
        'BRT-',
            'EUR=',
            'MIBG-DA1-ES',
            'TRNLTTFD1',
        'TRAPI2Mc1',
        'HT'

]

y_var = 'sscc_4'
regression_type = 'Linear'
significativas = False
prima = 0.9987
vols = [10000,20000,30000,40000,50000,50000,10000,10000,10000,10000,10000,10000] # This is a list of 12 elements, if the hedging is performed in more than 12 months (1 Year) the list should be expanded to the actual number of months 
n_vars = 4
window = 30
step_ahead_F1M = 1
step_ahead_F1Y = 12 # These step_aheads are just an example, all functions work with all step_aheads

initial_date = '2023-06-01'
final_date = '2023-12-01'



######### Use cases

### Perform hedging for the remaining months of the year

print('Perform hedging of the next 12 months. It starts on initial_date, and ends on final_date')
hedges = rw_hedging_forwards(df_c, x_cols, y_var, initial_date = initial_date, final_date = final_date,volumen = vols,  window = 30)
hedges.to_excel(f'hedges_{initial_date}_{final_date}.xlsx')




### Evaluate hedging F1M for 2022

initial_date = '2022-01-01'
final_date = '2022-12-01'

print('Evaluate hedging F1M')


d1 = rw_hedging_forwards_eval(df_c,x_cols, y_var,volumen = vols,initial_date = initial_date,final_date = final_date,regr_type =  regression_type, signif = significativas, num_variables = n_vars, window = window,  step_ahead = step_ahead_F1M) # Model hedging: Cash Flow
d1.to_excel(f'hedge_F1M_results_{initial_date}_{final_date}.xlsx') # Get the hedging results

# Comparison against the initial model of the variable Pool
d1 = d1.set_index('forward_date')

stat = d1[['cash_flow_inicial_EUR_MWh','cash_flow_EUR_MWh','cash_flow_prima_EUR_MWh']]

d1_pool = rw_hedging_forwards_eval(df_c,['POOL AVG'], y_var,volumen = vols,initial_date = initial_date,final_date = final_date,regr_type =  regression_type, window = window,  step_ahead = step_ahead_F1M)  # Pool based model hedging: Cash Flow sin coberturas
d1_pool = d1_pool.set_index('forward_date')

pool_df = d1_pool[['cash_flow_inicial_EUR_MWh']]
pool_df = pool_df.rename(columns={'cash_flow_inicial_EUR_MWh':'cash_flow_POOL_sin_coberturas_EUR_MWh'}) # The initial model of the variable: 'POOL' is considered the baseline for comparison


stat_pool = d1[['cash_flow_EUR_MWh','cash_flow_prima_EUR_MWh', 'Cuadrados_Con_C']]
stat_pool = pd.concat([stat_pool, pool_df['cash_flow_POOL_sin_coberturas_EUR_MWh']], axis=1)


m1 =  stat_pool['cash_flow_POOL_sin_coberturas_EUR_MWh'].mean()
m2 = stat_pool['cash_flow_EUR_MWh'].mean()
m3 =  stat_pool['cash_flow_prima_EUR_MWh'].mean()
m4 = 100 - 100 * (m2**2)/(m1**2)

results = []
results.append(('2022',d1['vars'], m1,m2,m3,m4))

df_results = pd.DataFrame(results, columns=['year', 'Variables','MEAN Cash Flow sin coberturas','MEAN Cash Flow EUR/MWh','MEAN Cash Flow Prima EUR/MWh','% Mejora'])

# Outputs

grafico_columnas_agrupadas(stat_pool[['cash_flow_EUR_MWh','cash_flow_POOL_sin_coberturas_EUR_MWh','cash_flow_prima_EUR_MWh']],'2022 F1M')
print(df_results)




### Evaluate hedging F1Y for 2022

print('Evaluate hedging F1Y')


d1 = rw_hedging_forwards_eval(df_c,x_cols, y_var,volumen = vols,initial_date = initial_date,final_date = final_date,regr_type =  regression_type, signif = significativas, num_variables = n_vars, window = window,  step_ahead = step_ahead_F1Y) # Model hedging: Cash Flow
d1.to_excel(f'hedge_F1Y_results_{initial_date}_{final_date}.xlsx') # Get the hedging results

# Comparison against the initial model of the variable Pool

d1 = d1.set_index('real_date') # The hedging results of each month are stored on the day the hedging was performed

d1_pool = rw_hedging_forwards_eval(df_c,['POOL AVG'], y_var,volumen = vols,initial_date = initial_date,final_date = final_date,regr_type =  regression_type, window = window,  step_ahead = step_ahead_F1Y)  # Pool based model hedging: Cash Flow sin coberturas
d1_pool = d1_pool.set_index('real_date')

pool_df = d1_pool[['cash_flow_inicial_EUR_MWh']]
pool_df = pool_df.rename(columns={'cash_flow_inicial_EUR_MWh':'cash_flow_POOL_sin_coberturas_EUR_MWh'}) # The initial model of the variable: 'POOL' is considered the baseline for comparison

stat_pool = d1[['cash_flow_EUR_MWh','cash_flow_inicial_EUR_MWh','cash_flow_prima_EUR_MWh']]
stat_pool = pd.concat([stat_pool, pool_df['cash_flow_POOL_sin_coberturas_EUR_MWh']], axis=1)


m1 =  stat_pool['cash_flow_POOL_sin_coberturas_EUR_MWh'].mean()
m2 = stat_pool['cash_flow_EUR_MWh'].mean()
m3 =  stat_pool['cash_flow_prima_EUR_MWh'].mean()
m4 = 100 - 100 * (m2**2)/(m1**2)
results = []
results.append(('2022',d1['vars'], m1,m2,m3,m4))

df_results = pd.DataFrame(results, columns=['year', 'Variables','MEAN Cash Flow sin coberturas','MEAN Cash Flow EUR/MWh','MEAN Cash Flow Prima EUR/MWh','% Mejora'])

# Outputs

grafico_columnas_agrupadas(stat_pool[['cash_flow_EUR_MWh','cash_flow_POOL_sin_coberturas_EUR_MWh','cash_flow_inicial_EUR_MWh','cash_flow_prima_EUR_MWh']],'2022 F1Y')
print(df_results)




