import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def grafico_columnas_agrupadas(df, title= None, eje:str = 'EUR/MWh'):
    '''
    Generate Cash flow main graph
    Input:
        df: Dataframe. Dataframe to plot
        title: String. Title of the graph
        eje: String. Axis name of the graph
    Output:
        The actual graph
    '''
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df_mes = df.copy()
    fechas = df_mes.index.tolist()

    fig, ax = plt.subplots(figsize=(22, 6))  

    colores = ['C0', 'C1', 'C2', 'C3', 'C4']
    ancho = 0.8 / len(df_mes.columns)  
    posiciones = np.arange(len(df_mes.columns)) * ancho

    for i, columna in enumerate(df_mes.columns):
        x = np.arange(len(fechas))
        y = df_mes[columna]
        ax.bar(x + posiciones[i], y, width=ancho, label=columna, color=colores[i])

    ax.set_title(f'Gráfico de cajas {title}')
    ax.set_xlabel('Fecha')
    ax.set_ylabel(eje)
    ax.legend()
    ax.set_xticks(np.arange(len(fechas)))
    ax.set_xticklabels([fecha.strftime('%Y-%m') for fecha in fechas], rotation=90)
    plt.subplots_adjust(bottom=0.2)
    ax.set_xlim(-ancho, len(fechas)  + ancho)

    plt.show()