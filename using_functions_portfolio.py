# Apartado de librerias necesarias para el procesamiento de datos 
import os 
os.chdir("/media/cristiandavid/CURSOS FEC/UNSA 2022 CC/semestre par/ECONOMIA/tesis/Optimum_portfolio_py/")
import classes 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import importlib
from scipy.stats import skew, kurtosis, chi2



importlib.reload(classes)
# Ahora importo los scripts necesarios, como classes y functions para cargar los 
# datos
#%% Corrigiendo el cargador de datos 

os.chdir("/media/cristiandavid/CURSOS FEC/UNSA 2022 CC/semestre par/ECONOMIA/tesis/Optimum_portfolio_py/acciones/")
asset = pd.read_csv('BBVA.MC.csv', usecols = ['Date'])
bench = pd.read_csv("^STOXX.csv", usecols = ['Date'])
asset = pd.to_datetime(asset.Date, dayfirst = True)
bench = pd.to_datetime(bench.Date, dayfirst = True)
asset['returns']

time = set(asset) & set(bench)

asset = asset[asset.isin(time)
asset.reset_index()





















#%% Funciones 
direccion = '/media/cristiandavid/CURSOS FEC/UNSA 2022 CC/semestre par/ECONOMIA/tesis/Optimum_portfolio_py/acciones/'
   



# def base_retornos(rics):
#     '''
#         Formato de leido CSV. 
#         Vector con el nombre de los archivos.csv
    
#         input: Vector de RICS 
#         return: Price df 
#                 Return df
        
#     '''
#     # Primero hallamos el menor 
#     menor = 100000
#     for ric in rics:
#         if (elementos(ric) < menor):
#             menor = elementos(ric)
#             ric_menor = ric
    
#     base_retornos = pd.DataFrame()
#     base_prices = pd.DataFrame()
#     menor_date = pd.read_csv(direccion + ric_menor)
#     menor_date['Date'] = pd.to_datetime(menor_date.Date, dayfirst = True)
#     base_retornos['Date'] = menor_date.Date
#     base_prices['Date'] = menor_date.Date
    
#     for ric in rics:         
#         data = pd.read_csv(direccion + ric).dropna(axis = 0)
#         data['Date'] = pd.to_datetime(data.Date, dayfirst = True)
#         times1 = list(data.Date.values)
#         times2 = list(data.Date.values)
#         common_times = list(set(times1) & set(times2))
#         data = data[data.Date.isin(common_times)]
#         data = data.reset_index()
#         base_retornos[ric.replace('.csv', '') + '_return'] = data.Close / data.Close.shift(1) - 1
#         base_prices[ric.replace('.csv', '') + '_price'] = data.Close   
        
#     return base_prices.dropna(axis = 0), base_retornos.dropna(axis = 0)

# def largo(acciones):
#     menor = 100000
#     for i in acciones:
#         data = pd.read_csv(direccion + i, usecols = ['Close']).dropna()
#         if data.Close.shape[0] < menor:
#             menor = data.Close.shape[0]
#             asset = i
#     asset_menor = pd.read_csv(direccion + asset, usecols = ['Close','Date']).dropna()
#     asset_menor.Date = pd.to_datetime(asset_menor.Date, dayfirst = True)
#     return set(asset_menor.Date)


def merge(acciones):
    data = pd.read_csv(direccion + acciones[1], usecols = ['Date'])
    data.Date = pd.to_datetime(data.Date, dayfirst = True)
    time = set(data.Date)
    for i in acciones:
        print(time)
        data = pd.read_csv(direccion + i, usecols = ['Date'])
        data.Date = pd.to_datetime(data.Date, dayfirst = True)
        time = set(data.Date) & time
    return time

def get_data(acciones):
    df = pd.DataFrame()
    time = merge(acciones)
    for i in acciones:
        data = pd.read_csv(direccion + i, usecols = ['Date', 'Close'])        
        data.Date = pd.to_datetime(data.Date, dayfirst = True)
        data = data[data.Date.isin(time)]
        data = data.reset_index()
        data['returns'] = data.Close/data.Close.shift() - 1
        df[i.replace('.csv','')] = data.returns
    return df
    


# def menor(acciones):
#     """
#     Inputs:
#         Acciones: iterable
#     """
#     for i in acciones:
        
            
#     data = pd.read_csv(direccion + sale, usecols = ['Date', 'Close']).dropna()
#     data.Date = pd.to_datetime(data.Date, dayfirst = True)
#     return data.Date
# df_p['Date'] = pd.to_datetime(df_p.Date, dayfirst = True)
# df_2p['Date'] = pd.to_datetime(df_2p.Date, dayfirst = True)
# times1 = list(df_p.Date.values)
# times2 = list(df_2p.Date.values)
# timess = list (set(times1) & set(times2))
# df_2p[df_2p.Date.isin(timess)]

#%%
# primero vere mis raw materials
os.chdir('/media/cristiandavid/CURSOS FEC/UNSA 2022 CC/semestre par/ECONOMIA/tesis/Optimum_portfolio_py/acciones/')
acciones = os.listdir()




# Creando la base de datos, tambien conocieda como la matriz de precios y retornos
# price_matrix, returns_matrix = base_retornos(acciones)
df = get_data(acciones)
df.isnull().sum()
plt.scatter(df['^STOXX'],df['SAN.MC'])
# Ahora veamos el enfoque del modelo CAPM
# Para lo cual necesitamos hacer una regresion lineal teniendo como variable 
# dependiente al vector de retornos de una accion y como independiente al vector 
# retornos de un activo de mercado. Como seria el caso de un indice de mercado
returns_matrix.columns
# Definamos el vector de retornos del activo de mercado
# Dado que tenemos acciones de mercados europeros tomaremos en cuenta al STOXX Y 
# STOXX50

# Definire una lista con los nombres de los activos de mercado
market_assets = ['^STOXX50E_return','^STOXX_return', '^S&P500_return']
x = np.array(returns_matrix[market_assets[2]])
# definidos mis activos de mercado, procedo a elegir un activo de este mercado 
y = np.array(returns_matrix['^VIX_return'])

# Instancio la clase linregress del modulo np.stat





