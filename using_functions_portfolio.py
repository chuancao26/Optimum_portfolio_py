# Apartado de librerias necesarias para el procesamiento de datos 
import os 
os.chdir('/media/cristiandavid/CURSOS FEC/UNSA 2022 CC/semestre par/ECONOMIA/tesis/Optimum_portfolio_py/')
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

#%% Funciones 

direccion = '/media/cristiandavid/CURSOS FEC/UNSA 2022 CC/semestre par/ECONOMIA/tesis/Optimum_portfolio_py/acciones/'

def date(ric):
    path = direccion + ric
    date_df = pd.read_csv(path)
    date_df = date_df.dropna() 
    return len(date_df['Date'])

def load_time_series(ric):
    # con su extension csv   
    # haciendo movil la entrada de datos
    path = direccion
    table_raw = pd.read_csv(path + ric)
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
    plt.tight_layout()
    plt.show()

def elementos(ric):
    data = pd.read_csv(direccion + ric).dropna(axis = 0)
    return data.shape[0]

def base_retornos(rics):
    '''
        input: Vector de RICS 
        return: Price df 
                Return df
        
    '''
    # Primero hallamos el menor 
    menor = 100000
    for ric in rics:
        if (elementos(ric) < menor):
            menor = elementos(ric)
            ric_menor = ric
    
    base_retornos = pd.DataFrame()
    base_prices = pd.DataFrame()
    menor_date = pd.read_csv(direccion + ric_menor)
    menor_date['Date'] = pd.to_datetime(menor_date.Date, dayfirst = True)
    base_retornos['Date'] = menor_date.Date
    base_prices['Date'] = menor_date.Date
    
    for ric in rics:         
        data = pd.read_csv(direccion + ric).dropna(axis = 0)
        data['Date'] = pd.to_datetime(data.Date, dayfirst = True)
        times1 = list(data.Date.values)
        times2 = list(data.Date.values)
        common_times = list(set(times1) & set(times2))
        data = data[data.Date.isin(common_times)]
        data = data.reset_index()
        base_retornos[ric.replace('.csv', '') + '_return'] = data.Close / data.Close.shift(1) - 1
        base_prices[ric.replace('.csv', '') + '_price'] = data.Close   
        
    return base_prices, base_retornos, ric_menor


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
# Primero cargemos los datos 
# usando las funciones presentes en el script de funciones 
ric = acciones[27]
date(ric)
ric2 = acciones [24]



# Hallando los retornos 
df_p = pd.read_csv('/media/cristiandavid/CURSOS FEC/UNSA 2022 CC/semestre par/ECONOMIA/tesis/Optimum_portfolio_py/acciones/' + ric)
df_2p = pd.read_csv('/media/cristiandavid/CURSOS FEC/UNSA 2022 CC/semestre par/ECONOMIA/tesis/Optimum_portfolio_py/acciones/' + ric2)
df_p.shape
df_2p.shape
df_p['Date'] = pd.to_datetime(df_p.Date, dayfirst = True)
df_2p['Date'] = pd.to_datetime(df_2p.Date, dayfirst = True)
times1 = list(df_p.Date.values)
times2 = list(df_2p.Date.values)
timess = list (set(times1) & set(times2))
df_2p[df_2p.Date.isin(timess)]


df_nuevo = pd.DataFrame()
df_nuevo['Retorno'] = df_p.Close / df_p.Close.shift(1) - 1
df_nuevo['Price'] = df_p.Close
# Podemos hacerlo al final

df_nuevo.Date
# df_nuevo.dropna(axis = 0, inplace = True)
df_nuevo.shape




# Graficos
plt.figure()
plt.plot(df_nuevo.Date, df_nuevo.Retorno)
plt.show()


plt.figure()
plt.plot(df_nuevo.Date, df_nuevo.Close)
plt.show()

# lo que queremos es juntar en un dataframe los retornos de todas las acciones en 
# un dataframe
# veamos la cantidad de elementos de cada uno de los archivos en la carpeta acciones




# veamos el proceso de filtracion para obtener un solo df con el date minimo
# t1 = t1[t1.Date.isin((set(t.Date) & set(t1.Date)))]





#%% Probando monstrito
prices_df, returns_df, menorss= base_retornos(acciones)
prices_df.isnull().sum()
returns_df.isnull().sum()
elementos(acciones[27])
prices_df[prices_df['^GDAXI_price'].isna()]






