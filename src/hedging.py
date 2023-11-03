import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

from src import config as conf
from src import aux_functions as aux
from src.modelization_functions import regression

###Perform Hedging

def rw_hedging_forwards(df:pd.DataFrame(), x_cols, y_var,volumen:list, initial_date:str ='2022-01-01', final_date:str ='2022-12-01', signif:bool = False,regr_type =  'Linear',num_variables:int = 4, window:int = 30):
    """ 
        Rolling window hedging. It gives the hedging swaps for the selected parameters, it outputs the df with all necessary hedge information for the selected period. 
        It takes the last date on the passed df as the last date of the training window
        Input:
            df: Dataframe. It takes a df with the objective function, and all spot and forward columns. The hedging is done for all dates in the index.
            x_cols: List of Spot columns. These are the columns that will be selected in the regression
            y_var: String of the name of the objettve fnction (y)
            volumen: List. Volume of SSCC for each month
            initial_date: String. It is the first month in which the hedging is done. It must have the format: 'YYYY-MM-01'
            final_date: String. It is the last month in which the hedging is done. It must have the format: 'YYYY-MM-01'
            signif: Boolean. True to apply the Step Wise method in the regression. False to select all variales passed in x_cols. Only used in the regression.
            regr_type: String. Linear, for linear regression, or Huber, for robust regression. Only used in the regression.
            num_variables: Integer. Maximum number of variables to select while doing Step Wise regression, it is applied when signif is True. Only used in the regression.
            window: Integer. Training window in which the regression is calibrated
        Outputs: Dataframe.
            Dataframe with the swaps for each month and variable, it has additional columns, to provide more information:
                SSCC estimado: SSCC estimated for that month
                Volumen: Volume for each month.
                Variables: Variable used in the regression 
                Coeficientes: Coefficients of the regression
                Forwards: Forward of eah variable for the selected month ( The month in which the hedginig performed)
                Accion: Action to perform. 'Comprar' if the coefficient of a variable is positive and 'Vender' if the coefficient of a variable is negative
                Swaps: Actual volume to buy or sell
    """
    df_total = pd.DataFrame()

    initial_date = datetime.strptime(initial_date, '%Y-%m-%d')
    final_date = datetime.strptime(final_date, '%Y-%m-%d')
    step_ahead = (final_date.year - initial_date.year) * 12 + final_date.month - initial_date.month

    ###### Date range interval for the train dataset, delimited by the window size

    date = df.index[-1]  - relativedelta(months = window)               # Starting date of the training window
    date_max = df.index[-1]                                             # End date of the training window. Month in which I perform the hedging
    date_max_t = final_date                                             # Maximum date over which I want to make hedges
    unique_dates = pd.date_range(start = initial_date, end = date_max_t,freq = 'MS')

    if initial_date <= date_max:
        print('The max date of the passed dataframe should not be prior to the initial date passed. The initial date has already passed')

    df_out2 = df.loc[date : date_max] # Dataframe with training window
    
    x1 = x_cols.copy()
    try:
        x1.remove('HT')
    except:
        pass

    df_forwards = aux.create_forward_df(df,cols = x1, date_ini = str(date_max),date_end = date_max_t) # Test dataframe: Forwards
    df_forwards['HT'] = [df['HT_f'].loc[date_max:date_max][0]] * len(df_forwards)

    for step in range(0, step_ahead + 1 ):  # Iteration on each month of the test window delimited by the step_ahead parameter
        vol_index = step -1

        date_max_step = unique_dates[step ].date() # Date of each step
        
        ###### Dataframe definition for each step
        
        df_test = df_forwards.loc[date_max_step : date_max_step]

        ###### Regression with the forward values
        
        df_reg = regression(df_out2, x_cols, y_var, df_test = df_test, reg_type =  regr_type, significativas= signif, n_vars = num_variables)
        res_pred = np.concatenate([np.ravel(rr) for rr in df_reg['pred']])
        
        ###### Calculate liquidations: LIQUIDATIONS

        swaps = []
        dates = []
        fws = []
        vars = df_reg['vars'][0]
        coefs = df_reg['coef'][0]

        ###### Swap liquidations: SWAPS
        
        for numero,c in enumerate(coefs):
            
            factor = c * volumen[vol_index]
            if vars[numero] == 'TRAPI2Mc1' or vars[numero] == 'BRT-': # Cotizan en dolares
                fw = df_test[vars[numero]].loc[date_max_step : date_max_t][0] / df_test['EUR='].loc[date_max_step : date_max_t][0]
                swap = float(factor * ( -  fw   ) ) # swap = factor * ( - [forward del M +1 en le mes M]  )
            elif vars[numero] == 'HT':
                fw = df_test[vars[numero]].loc[date_max_step : date_max_t][0]
                swap = 0
            else:
                fw = df_test[vars[numero]].loc[date_max_step : date_max_t][0]
                swap = float(factor *(  -  fw   ) ) # swap = factor * ( - [forward del M +1 en el mes M]  )

            swaps.append(swap)
            dates.append(date_max_step)
            fws.append(fw)

        dd = pd.DataFrame({'SSCC estimado': res_pred,'Volumen':volumen[vol_index] ,'Variables':[vars],'Coeficientes':[coefs],'Forwards':[fws],'Accion': [0],'Swaps': [swaps]}, index=sorted(set(dates)))
        df_total = pd.concat([df_total, dd], axis=0)
    
    df_final = df_total.explode(['Variables','Coeficientes','Forwards','Swaps'])
    df_final['Accion'] = np.where( df_final['Swaps'] > 0 , 'Comprar', np.where( df_final['Swaps'] < 0 , 'Vender', np.where( df_final['Swaps'] == 0 , 'NO SE CUBRE', 0 )) )

    return df_final


# Evaluate Hedging

def rw_hedging_forwards_eval(df:pd.DataFrame(), x_cols, y_var,volumen:list, initial_date:str ='2022-01-01', final_date:str ='2022-12-01', signif:bool = False, prima:float =  0.9987, regr_type =  'Linear',num_variables:int = 4, window:int = 30, step_ahead:int = 1):

    """ 
    Rolling window hedging. It evaluates the hedging for the selected parameters, it outputs the cash flow for the selected period
    Input:
        df: Dataframe. It takes a df with the objective function, and all spot and forward columns. The hedging is done for all dates in the index.
        x_cols: List of Spot columns. These are the columns that will be selected in the regression
        y_var: String of the name of the objettve fnction (y)
        volumen: List. Volume of SSCC for each month
        initial_date: String. It is the first month in which the hedging is done. It must have the format: 'YYYY-MM-01'
        final_date: String. It is the last month in which the hedging is done. It must have the format: 'YYYY-MM-01'
        signif: Boolean. True to apply the Step Wise method in the regression. False to select all variales passed in x_cols. Only used in the regression.
        prima: Float. Selected adder (Prima). The prima is added in the final calulations as an addition to the base cash flow
        regr_type: String. Linear, for linear regression, or Huber, for robust regression. Only used in the regression.
        num_variables: Integer. Maximum number of variables to select while doing Step Wise regression, it is applied when signif is True. Only used in the regression.
        window: Integer. Training window in which the regression is calibrated
        step_ahead: Integer. Number of months ahead selected to perform hedging in. If step_ahead = 1 it means that the hedging is caculated for 1 month ahead
    Outputs: Dataframe.
        Dataframe with the final hedging calculations for each month, it has the following columns:
            vars: List of the variables used
            coefs: List of the coefficients used in the regression
            real_date: Date (Month) in which the hedging is done (m)
            forward_date: Date (Month) for which the hedging is done
            sscc_estimado: Estimated value of the SSCC of the months in the test dataframe
            sscc_spot_m1: Real value of the SSCC of the months in the test dataframe
            total_liquid: Sum of all liquidations done by the variables
            r2: R2 of the regression. It is the R2 in-sample.
            cash_flow_EUR: It is the Cash Flow resulting of the entire hedging process
            cash_flow_prima_EUR: It is the Cash Flow resulting of the entire hedging process plus an adder
            cash_flow_inicial: It is the Cash Flow resulting of the initial part of the process. It does not take into account the swap liquidations
            cash_flow_EUR_MWh: cash_flow_EUR divided by the volume (volumen)
            cash_flow_prima_EUR_MWh: cash_flow_prima_EUR_MWh divided by the volume (volumen)
            cash_flow_inicial_EUR_MWh: cash_flow_inicial divided by the volume (volumen)
            Cuadrados_Sin_C: It is used for the %Mejora metric, it is cash_flow_inicial_EUR_MWh ^2
            Cuadrados_Con_C: It is used for the %Mejora metric, it is cash_flow_EUR ^2

    """
    
    df_total = pd.DataFrame()

    d =  df.loc[initial_date:].index[0] - relativedelta(months = window) # Date defiition
    df = df.loc[ d :final_date ]

    unique_dates = df.index.unique() # List of dates of the DataFrame
    unique_dates1 = unique_dates[:-(window)-(step_ahead) + 1] # List of dates to iterate over

    for idx,i in enumerate(unique_dates1):

        ###### Date range interval for the train dataset, delimited by the window size

        date = i.date()                                                  # Starting date of the training window
        date_max = unique_dates[idx + window -1].date()                  # End date of the training window. Month in which I perform the hedging
        date_max_t = unique_dates[idx + window +  step_ahead -1 ].date() # Maximum date over which I want to make coverages

        df_out2 = df.loc[date : date_max] # Dataframe with training window
        x1 = x_cols.copy()
        try:
            x1.remove('HT')
        except:
            pass
        df_forwards = aux.create_forward_df(df,cols = x1, date_ini = str(date_max),date_end = date_max_t) # Dataframe de test: Forwards
        df_forwards['HT'] = [df['HT_f'].loc[date_max:date_max][0]] * len(df_forwards)
    
        for step in range(1, step_ahead+1 ):  # Iteration on each month of the test window delimited by the step_ahead parameter
            vol_index = step -1
            df_res  = pd.DataFrame()

            date_max_step = unique_dates[idx + window + step -1 ].date() # Date of each step

            ###### Dataframe definition for each step
            
            df_obj = df[y_var].loc[date_max_step : date_max_step]
            df_forwards2 = df_forwards.loc[date_max_step : date_max_step]

            df_test = df_forwards2.join(df_obj, how = 'left')
        

            ###### Regression with the forward values
            
            df_reg = regression(df_out2, x_cols, y_var, df_test = df_test, reg_type =  regr_type, significativas= signif, n_vars = num_variables)
            
            ###### Calculate liquidations: LIQUIDATIONS

            liquid = []
            vars = df_reg['vars'][0]
            coefs = df_reg['coef'][0]


            ###### Swap liquidations: SWAPS
            
            for numero,c in enumerate(coefs):
               
                factor = c * volumen[vol_index]
                if vars[numero] == 'TRAPI2Mc1' or vars[numero] == 'BRT-': # Cotizan en dolares
                    
                    swap = float(factor * ( df[vars[numero]].loc[date_max_step : date_max_t][0] / df['EUR='].loc[date_max_step : date_max_t][0] -  df_forwards2[vars[numero]].loc[date_max_step : date_max_t][0] / df_forwards2['EUR='].loc[date_max_step : date_max_t][0]   ) ) # swap = factor * ( [spot M +1] - [forward M +1 in M]  )
                elif vars[numero] == 'HT':
                    swap = 0
                else:
                    
                    swap = float(factor *( df[vars[numero]].loc[date_max_step : date_max_t][0] -  df_forwards2[vars[numero]].loc[date_max_step : date_max_t][0]   ) ) # swap = factor * ( [spot M +1] - [forward M +1 in M]  )

                liquid.append(swap)
                    
            res_pred = np.concatenate([np.ravel(rr) for rr in df_reg['pred']])

            ###### CALCULATIONS

            df_res['vars'] = [vars]
            df_res['coefs'] = [coefs]
            df_res['real_date'] = date_max
            df_res['forward_date'] = date_max_step
            df_res['sscc_estimado'] = res_pred[0]
            df_res['sscc_spot_m1'] = float(df_obj[0])
            df_res['total_liquid'] = sum(liquid)
            df_res['r2'] = df_reg['r2'][0]

            df_res['cash_flow_EUR'] = volumen[vol_index] * (df_res['sscc_estimado'] - df_res['sscc_spot_m1'] ) + df_res['total_liquid']
            df_res['cash_flow_prima_EUR'] = volumen[vol_index] * (prima + df_res['sscc_estimado'] - df_res['sscc_spot_m1'] ) + df_res['total_liquid']
            df_res['cash_flow_inicial'] = volumen[vol_index] * (df_res['sscc_estimado'] - df_res['sscc_spot_m1'] )
            
            df_res['cash_flow_EUR_MWh'] = df_res['cash_flow_EUR'] / volumen[vol_index]
            df_res['cash_flow_prima_EUR_MWh'] = df_res['cash_flow_prima_EUR'] / volumen[vol_index]
            df_res['cash_flow_inicial_EUR_MWh'] = (df_res['sscc_estimado'] - df_res['sscc_spot_m1'] )

            df_res['Cuadrados_Sin_C'] = df_res['cash_flow_inicial_EUR_MWh']**2
            df_res['Cuadrados_Con_C'] = df_res['cash_flow_EUR_MWh']**2
             
            df_total = pd.concat([df_total, df_res], axis=0)

    return df_total.reset_index(drop = True)