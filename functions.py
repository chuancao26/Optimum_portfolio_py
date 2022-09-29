


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import importlib
from scipy.stats import skew, kurtosis, chi2

import os 

os.chdir('E:/UNSA 2022 CC/semestre par/tesis/acciones')




def load_time_series(ric, file_extension = 'csv'):
    # con su extension csv   
    # haciendo movil la entrada de datos
    
    path = 'E:/UNSA 2022 CC/semestre par/tesis/acciones/'
    table_raw = pd.read_csv(path + ric + '.' + file_extension)
    # ahora convertiremos estos dates simples en datetimes
    # para lo q creare un df
    t = pd.DataFrame()
    t['Date'] = pd.to_datetime(table_raw['Date'], dayfirst = True)
    t['close'] = table_raw['Close']
    # Hallando los retornos 
    t['close_previous'] = table_raw.Close.shift(1)
    t['return_close'] = t.close/t.close_previous - 1
    t = t.dropna()      
    t = t.reset_index(drop = True)
    # reciclano el codigo para las metricas estadisticas
    # Eliminando los na
    x = t.return_close.values
    x_str = 'Real Return of ' + ric
    # x_size = len(x)
    return x, x_str, t

def time_series_graph_of(t, ric):
    plt.figure()
    plt.plot(t.Date, t.close, color = 'r')
    plt.title('Time Series Graph of ' + ric)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()
    
def distributions_graph_of(x, ric, string_plot, bins = 100):
    plt.figure()
    plt.hist(x, bins)
    plt.title('Grafico de distribucion para la variable ' + ric)
    plt.xlabel(string_plot)
    plt.show()
