# SSCC file path

sscc_path = './data/liquicomun.xlsx'


# HT file path:

path_ht = './data/PBF_2014_2023.xlsx'


# Dict for the commodities files with data extracted form the Reuters webpage

commodities_files = {'OMEL': './data/OMEL.xlsx',
                     'OMIP': './data/OMEL.xlsx',
                     'EURUSD': './data/Reikon_commodities.xlsx',
                     'EUA': './data/Reikon_commodities.xlsx',
                     'BRENT': './data/Reikon_commodities.xlsx',
                     'API2': './data/Reikon_commodities.xlsx',
                     'TTF': './data/Reikon_commodities.xlsx',
                     'MIBGAS PVB': './data/Reikon_commodities.xlsx'
            }

# Columns to select from the SSCC

columns = [
         'Restricciones técnicas PDBF',
         'Banda de regulación secundaria',
         'Reserva de potencia adicional a subir',
         'Restricciones técnicas en tiempo real',
         'Incumplimiento de energía de balance', 
         'Coste desvíos',
         'Saldo desvíos', 
         'Control del factor de potencia', 
         'Saldo PO 14.6',
         'Servicios de ajuste',
         'Servicio de interrumpibilidad'
         ]


# List with all SPOT variables in the commodities file to consider

spot_vars = [
      'POOL AVG',
      'EUR=',
      'CFI2c5', # It takes only the december forward
      'BRT-',
      'TRAPI2Mc1',
      'TRNLTTFD1',
      'MIBG-DA1-ES'
      ]


# Objective variable definition: sscc_4, sscc_8

sscc_4 = ['Restricciones técnicas PDBF', 'Banda de regulación secundaria',
       'Reserva de potencia adicional a subir',
       'Restricciones técnicas en tiempo real']

sscc_8 = ['Restricciones técnicas PDBF', 'Banda de regulación secundaria',
       'Reserva de potencia adicional a subir',
       'Restricciones técnicas en tiempo real','Saldo desvíos',
       'Incumplimiento de energía de balance','Control del factor de potencia', 'Saldo PO 14.6']



# Forward variables definition:

forward_1m = {
            'POOL AVG': 'OMIPFTBMc1',
            'EUR=': 'EUR1MV=',
            'CFI2c5': 'CFI2c9',
            'BRT-': 'LCOc1',
            'TRAPI2Mc1': 'TRAPI2Mc2',
            'TRNLTTFD1': 'TRNLTTFMc1',
            'MIBG-DA1-ES': 'MIBGMESMc1'
}

forward_2m = {
            'POOL AVG': 'OMIPFTBMc2',
            'EUR=': 'EUR2MV=',
            'CFI2c5': 'CFI2c9',
            'BRT-': 'LCOc2',
            'TRAPI2Mc1': 'TRAPI2Mc3',
            'TRNLTTFD1': 'TRNLTTFMc2',
            'MIBG-DA1-ES': 'MIBGMESMc2'
}

forward_3m = {
            'POOL AVG': 'OMIPFTBMc3',
            'EUR=': 'EUR3MV=',
            'CFI2c5': 'CFI2c9',
            'BRT-': 'LCOc3',
            'TRAPI2Mc1': 'TRAPI2Qc1',
            'TRNLTTFD1': 'TRNLTTFMc3',
            'MIBG-DA1-ES': 'MIBGMESQc1'
}

forward_4m = {
            'POOL AVG': 'OMIPFTBMc4',
            'EUR=': 'EUR4MV=',
            'CFI2c5': 'CFI2c9',
            'BRT-': 'LCOc4',
            'TRAPI2Mc1': 'TRAPI2Qc2',
            'TRNLTTFD1': 'TRNLTTFMc4',
            'MIBG-DA1-ES': 'MIBGMESQc2'
}

forward_5m = {
            'POOL AVG': 'OMIPFTBMc5',
            'EUR=': 'EUR5MV=',
            'CFI2c5': 'CFI2c9',
            'BRT-': 'LCOc5',
            'TRAPI2Mc1': 'TRAPI2Qc2',
            'TRNLTTFD1': 'TRNLTTFQc2',
            'MIBG-DA1-ES': 'MIBGMESQc2'
}

forward_6m = {
            'POOL AVG': 'OMIPFTBMc6',
            'EUR=': 'EUR6MV=',
            'CFI2c5': 'CFI2c9',
            'BRT-': 'LCOc6',
            'TRAPI2Mc1': 'TRAPI2Qc2',
            'TRNLTTFD1': 'TRNLTTFQc2',
            'MIBG-DA1-ES': 'MIBGMESQc2'
}

forward_7m = {
            'POOL AVG': 'OMIPFTBQc3',
            'EUR=': 'EUR7MV=',
            'CFI2c5': 'CFI2c9',
            'BRT-': 'LCOc7',
            'TRAPI2Mc1': 'TRAPI2Qc3',
            'TRNLTTFD1': 'TRNLTTFQc3',
            'MIBG-DA1-ES': 'MIBGMESQc3'
}
forward_8m = {
            'POOL AVG': 'OMIPFTBQc3',
            'EUR=': 'EUR8MV=',
            'CFI2c5': 'CFI2c9',
            'BRT-': 'LCOc8',
            'TRAPI2Mc1': 'TRAPI2Qc3',
            'TRNLTTFD1': 'TRNLTTFQc3',
            'MIBG-DA1-ES': 'MIBGMESQc3'
}
forward_9m = {
            'POOL AVG': 'OMIPFTBQc3',
            'EUR=': 'EUR9MV=',
            'CFI2c5': 'CFI2c9',
            'BRT-': 'LCOc9',
            'TRAPI2Mc1': 'TRAPI2Qc3',
            'TRNLTTFD1': 'TRNLTTFQc3',
            'MIBG-DA1-ES': 'MIBGMESQc3'
}
forward_10m = {
            'POOL AVG': 'OMIPFTBQc4',
            'EUR=': 'EUR10MV=',
            'CFI2c5': 'CFI2c9',
            'BRT-': 'LCOc10',
            'TRAPI2Mc1': 'TRAPI2Qc4',
            'TRNLTTFD1': 'TRNLTTFQc4',
            'MIBG-DA1-ES': 'MIBGMESYc1'
}
forward_11m = {
            'POOL AVG': 'OMIPFTBQc4',
            'EUR=': 'EUR11MV=',
            'CFI2c5': 'CFI2c9',
            'BRT-': 'LCOc11',
            'TRAPI2Mc1': 'TRAPI2Qc4',
            'TRNLTTFD1': 'TRNLTTFQc4',
            'MIBG-DA1-ES': 'MIBGMESYc1'
}
forward_12m = {
            'POOL AVG': 'OMIPFTBQc4',
            'EUR=': 'EUR1YV=',
            'CFI2c5': 'CFI2c9',
            'BRT-': 'LCOc12',
            'TRAPI2Mc1': 'TRAPI2Qc4',
            'TRNLTTFD1': 'TRNLTTFQc4',
            'MIBG-DA1-ES': 'MIBGMESYc1'
}