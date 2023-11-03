import pandas as pd
import numpy as np
from src import config as conf

from dateutil.relativedelta import relativedelta

# Preprocessing

def add_new_vars(df: pd.DataFrame(),spot_cols:list):
    '''
    It takes a df of monthly data and it computes the synthetic variables of the original vars.
    601: For each month, it is the average of the previous 6 months
    603: For each month, it is the average of the previous 2 quarters.
    Input:
        df: Dataframe. Dataframe of monthy data
        spot_cols: List. Column names of the df that are wanted to calculate the synthetic vars

    Output:
        It outputs the same df with the extra columns
    '''
    for var in spot_cols:
        df1 = df.rename_axis('index').reset_index() 
        df1 = df1.groupby(df1['index'].dt.to_period('Q')).mean()
        df1[f'{var}603'] = df1[var].rolling(window=2, min_periods=1).mean().shift(1)
        df1 = df1.resample('M', label='right').ffill()
        df1.index = df1.index.to_timestamp(how = 'S')

        df2 = df.rename_axis('index').reset_index() 
        df2 = df2.groupby(df2['index'].dt.to_period('M')).mean()
        df2[f'{var}601'] = df2[var].rolling(window=6, min_periods=1).mean().shift(1)
        df2.index = df2.index.to_timestamp(how = 'S')


        df = df.join(df2[[f'{var}601']], how ='left')
        df = df.join(df1[[f'{var}603']], how ='left')

    return df.ffill()


## Hedging related functions

def diferencia_trimestres(df, fecha_fija):
    '''
    It calculate the quarter's difference between the fixed date (fecha_fija) and the rest of the dates
    Input:
        df: Dataframe. Dataframe with monthly data and date index
        fecha_fija: Datetime. Datetime date to take as reference

    Output:
        New dataframe with an extra column with the quarter's difference
    '''
    df1 = df.copy()
    
    trimestre_fijo = (fecha_fija.month - 1) // 3 + 1 # Calculate the quarter of the fixed date
    año_fijo = fecha_fija.year
    df1['q'] = ((df1.index.year - año_fijo) * 4) + (df1.index.quarter - trimestre_fijo)

    return df1


def create_forward_df(df, cols:list, date_ini,date_end):
    '''
    It extracts the forwards based on dicts defined in the config.py file where the specific columns are mapped for each month.
    The forward value of the next month (m+1), it is indexed in the m+1 date, this logic applies for all forwards
    Input:
        df: Dataframe with monthly data which has all forward variables
        Date_ini: String. It is the the date in which you are performing the hedging (m), the hedging will start in m+1 until date_end
        Date_end: String. It is the last forward date, for intervals longer than 12 months, for each month it will take the F1Y (Forward of the next year)

    Output:
        It outputs a different df with the forward values in columns named like the spot ones.
        the df has the date rage defined in date_ini + 1 and date_end, the hedging date interval.
    '''
    
    dates = pd.date_range(start=date_ini, end=date_end, freq='MS')

    dict_keys = {key: conf.forward_1m[key] for key in cols}
    df2 = pd.DataFrame(columns = dict_keys.keys(),index=dates)
    df3 = pd.DataFrame()

    for idx,i in zip(range(len(dates) - 1), dates):
                  
        if idx < 12:
            cols_dict = f'conf.forward_{idx+1}m'

            df1 = df.loc[date_ini]
            df1 = df1[eval(cols_dict).values()]
            
            df1 = pd.DataFrame(df1).T
            df1.index = df1.index + pd.DateOffset(months=1+idx) # It takes the first forward month 
            for key, value in eval(cols_dict).items():
                df2[key] = df1[value]
        
        else: # If the date range is more than a year, for each month it takes the 1Y forward
            cols_dict = f'conf.forward_{12}m'
            df1 = df.loc[date_ini]
            df1 = df1[eval(cols_dict).values()]
        
            df1 = pd.DataFrame(df1).T
            df1.index = df1.index + pd.DateOffset(months=1+idx) # It takes the first forward month 
            for key, value in eval(cols_dict).items():
                df2[key] = df1[value]
                
        df3 = pd.concat([df3,df2]).dropna()

    return df3



def create_forward_df_60(df,df_forward, vars_60:list, date_ini,date_end):
    '''
    It is based on the create_forward_df function, and it calculates the forwards of any 601 or 603 variable passed 
    The forward value of the next month (m+1), it is indexed in the m+1 date, this logic applies for all forwards
    Input:
        df: Dataframe. Dataframe with monthly data which has all forward variables
        vars60: List. List of 601 and 603 columns 
        Date_ini: String. is the the date in which you are performing the hedging (m), the hedging will start in m+1 until date_end
        Date_end: String. It is the last forward date, for intervals longer than 12 months, for each month it will take the F1Y (Forward of the next year)

    Output:
        It outputs a different df with the forward values in columns named like the spot ones.
        the df has the date rage defined in date_ini + 1 and date_end, the hedging date interval.
    '''

    d = df.loc[date_ini:date_ini].index[0]  - relativedelta(months = 9)
    d1 = pd.concat([df.loc[d:date_ini], df_forward.loc[date_ini:date_end]], ignore_index=False)
    for var_brent in vars_60:
        if '603' in var_brent:
            var_brent1 = var_brent[:-3]

            df1 = d1[var_brent1].rename_axis('index').reset_index() 
            df1 = df1.groupby(df1['index'].dt.to_period('Q')).mean()
            df1[f'{var_brent1}603'] = df1[var_brent1].rolling(window=2, min_periods=1).mean().shift(1)
            df1 = df1.resample('M', label='right').ffill()
            df1.index = df1.index.to_timestamp(how = 'S')
            df_forward = df_forward.join(df1[[f'{var_brent1}603']], how ='left')


        if '601' in var_brent:
            
            var_brent1 = var_brent[:-3]
            df2 = d1[var_brent1].rename_axis('index').reset_index() 
            df2 = df2.groupby(df2['index'].dt.to_period('M')).mean()
            df2[f'{var_brent1}601'] = df2[var_brent1].rolling(window=6, min_periods=1).mean().shift(1)
            df2.index = df2.index.to_timestamp(how = 'S')
            
            df_forward = df_forward.join(df2[[f'{var_brent1}601']], how ='left')

    return df_forward