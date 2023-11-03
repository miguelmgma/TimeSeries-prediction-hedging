import pandas as pd
from src import config as conf

def read_new_sscc(path: str):
    '''
    Excel file that contains all data regarding the SSCC: Liquicomun.xlsx
    Input:
        path: String. Path to the file
        temp: String. Temporality desired on the data. Defaults to Daily.
    Output:
        Dataframe with the correct columns and concepts of the SSCC
        Minimum date of the dataframe
        Maximum date of the dataframe
    '''
    df_l = pd.read_excel(path).set_index('fecha')
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    df_l = df_l.select_dtypes(include=numerics)
    df_l = df_l.resample("MS").mean()

    df_l['Restricciones técnicas PDBF'] = df_l['RT3'] + df_l['CT3']
    df_l['Banda de regulación secundaria'] = df_l['BS3']
    df_l['Reserva de potencia adicional a subir'] = df_l['PS3']
    df_l['Restricciones técnicas en tiempo real'] = df_l['RT6']
    df_l['Incumplimiento de energía de balance'] = df_l['BALX']
    df_l['Saldo desvíos'] = df_l['EXD']
    df_l['Control del factor de potencia'] =  df_l['CFP']
    df_l['Saldo PO 14.6'] = df_l['IN7']
    

    return df_l[['Restricciones técnicas PDBF', 'Banda de regulación secundaria',
       'Reserva de potencia adicional a subir',
       'Restricciones técnicas en tiempo real','Saldo desvíos',
       'Incumplimiento de energía de balance','Control del factor de potencia', 'Saldo PO 14.6']],df_l.index[0],df_l.index[-1]

## SSCC commodities files
def read_commodities(commodities_dict: dict, start_date:str, max_date:str, temp: str = 'Daily'):
    '''
    Function that takes a dict containing the name and path to each commodity and creates a joined dataframe with all data
    Input:
    Commodity dictionary containing as keys the name of each concept
        commodities_dict: Dictionary. Dictionary with the name and paths of the commodities
        start_date: String. Minimum date of the dataframe
        max_date: String. Maximum date of the dataframe
        temp: String. Temporality desired on the data. Defaults to Daily.
    Output: Dataframe.
        Dataframe with the all commodities data, resampled by the temp parameter
    '''
        
    # Set the date range

    d = pd.Period(max_date,freq='M').end_time.date()
    df_range= pd.DataFrame(pd.date_range(start_date, str(d), freq='D'))
    df_range.columns = ['Date']
    df_range = df_range.set_index('Date')
    for k,v in commodities_dict.items():

        
        df = pd.read_excel(v, sheet_name=k)

        if k == 'OMEL':
            df.columns = list(df.iloc[4]) 
            df = df.iloc[5:].reset_index(drop = True)
            df = df.rename(columns = {'Fecha correcta': 'Date','Media POOL':'POOL AVG'})
            df = df[['POOL AVG', 'Date']].set_index('Date')
            df = df.astype(float)

        if k == 'OMIP': 
            df.columns = df.iloc[3]
            df = df.iloc[5:,1:]
            df.columns = ['Date', *df.columns[1:]]
            df = df.set_index(df.columns[0])
            df = df.bfill() # Impute all NAN values with the next non-NAN 
            df = df.astype(float)

        if k == 'EURUSD' or k == 'BRENT' or k == 'EUA' or k == 'API2' or  k == 'TTF'  or k == 'MIBGAS PVB':
            df.columns = list(df.iloc[2])
            df = df.iloc[5:,1:].reset_index(drop = True)
            df.columns = ['Date', *df.columns[1:]]
            df = df.set_index('Date')
            df = df.bfill() # Impute all NAN values with the next non-NAN 
            df = df.astype(float)

        df_range = pd.concat([df_range, df], axis=0)
        
    if temp == 'Monthly':
        return df_range.resample("MS").mean().loc[:max_date]
    if temp == 'Weekly':
        return df_range.resample("W").mean().loc[:max_date]
    else:
        return  df_range.resample("MS").mean().loc[:max_date]

    
def read_ht(path:str, df:pd.DataFrame()):
    '''
    This funciton reads the HT file. It imports it and sum all columns considered to be HT: Carbon, Fuelgas, CC and Cogen.
    Also, for out-of-sample regressions it takes the mean of the previous 24 values.
    It creates 2 new columns in the passed Dataframe.
    Input:
        path: String. File path
        df: Dataframe. Dataframe to which the HT columns are attached
    Output:
        The passed Dataframe has 2 additional columns:
        HT: Real HT value for each month
        HT_f: 'Forward' values, they are the mean of the previous 24 values. Used in out_of_sample regression like hedging
    '''
    ht = pd.read_excel(path)
    ht1 = ht.set_index('Fecha').resample('MS').mean().fillna(0)
    ht1['HT'] =  ht1[ 'CARBON']+ ht1[ 'FUELGAS'] + ht1['CC'] + ht1['COGEN']
    ht1['HT_f'] = ht1['HT'].rolling(24).mean()
    ht1 = ht1[['HT','HT_f']]
    return df.join(ht1, how = 'left')

def df_preparation():
    '''
    Function that creates the objective dataframe
    Output:
        Dataframe with the all combined data
    '''
    df_sscc,start_date, max_date = read_new_sscc(conf.sscc_path) # Read SSCC
    df_sscc.index = pd.to_datetime(df_sscc.index)
    df_sscc['sscc_4'] = df_sscc[conf.sscc_4].sum(axis=1) # Define SSCC

    df_comm = read_commodities(conf.commodities_files,start_date = start_date,  max_date = max_date,temp = 'Monthly') # Read commodities
    df_comm.index = pd.to_datetime(df_comm.index)
    df1 = df_sscc[['sscc_4']].join(df_comm, how = 'left').fillna(0)

    try:
        df1 = read_ht(conf.path_ht,df1)
    except:
        pass
    return df1