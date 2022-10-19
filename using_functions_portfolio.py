# Apartado de librerias necesarias para el procesamiento de datos 
import os 
os.chdir("E:/UNSA 2022 CC/semestre par/ECONOMIA/tesis/Optimum_portfolio_py")
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

direccion = 'E:/UNSA 2022 CC/semestre par/ECONOMIA/tesis/Optimum_portfolio_py/acciones/'

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

def time_series_graph_of(x, y, name = ''):
    plt.figure()
    plt.plot(x, y, color = 'r')
    plt.title('Time Series Graph of ' + name)
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
        Formato de leido CSV. 
        Vector con el nombre de los archivos.csv
    
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
        
    return base_prices.dropna(axis = 0), base_retornos.dropna(axis = 0)


# df_p['Date'] = pd.to_datetime(df_p.Date, dayfirst = True)
# df_2p['Date'] = pd.to_datetime(df_2p.Date, dayfirst = True)
# times1 = list(df_p.Date.values)
# times2 = list(df_2p.Date.values)
# timess = list (set(times1) & set(times2))
# df_2p[df_2p.Date.isin(timess)]



#%%
# primero vere mis raw materials
os.chdir('E:/UNSA 2022 CC/semestre par/ECONOMIA/tesis/Optimum_portfolio_py/acciones/')
acciones = os.listdir()

# Creando la base de datos, tambien conocieda como la matriz de precios y retornos
price_matrix, returns_matrix = base_retornos(acciones)

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

# Instancio la clase linregress del modulo np.stats
capm = scipy.stats.linregress(x,y)
dir(capm)
capm.slope
capm.intercept


plt.figure()
plt.scatter(x,y)
plt.show()

plt.hist(y, bins = 100)
plt.show()
# (1 + np.mean(x))**360 - 1 
# /

























