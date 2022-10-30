# Apartado de librerias necesarias para el procesamiento de datos 
import os 
os.chdir("/home/cristiandavid/Documents/unsa_2022/economia/tesis_2/Optimum_portfolio_py/")
import classes 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import importlib
from scipy.stats import skew, kurtosis, chi2
import yfinance as yf



importlib.reload(classes)
# Ahora importo los scripts necesarios, como classes y functions para cargar los 
# datos
#%% Corrigiendo el cargador de datos 

os.chdir("/home/cristiandavid/Documents/unsa_2022/economia/tesis_2/Optimum_portfolio_py/acciones/")
asset = pd.read_csv('BBVA.MC.csv', usecols = ['Date'])
bench = pd.read_csv("^STOXX.csv", usecols = ['Date'])
asset = pd.to_datetime(asset.Date, dayfirst = True)
bench = pd.to_datetime(bench.Date, dayfirst = True)
asset['returns']

time = set(asset) & set(bench)

asset = asset[asset.isin(time)
# asset.reset_index()

#%% Funciones 
direccion = '/home/cristiandavid/Documents/unsa_2022/economia/tesis_2/Optimum_portfolio_py/acciones/'
def yahoo_api(etfs):
    os.chdir('/home/cristiandavid/Documents/unsa_2022/economia/tesis_2/Optimum_portfolio_py/acciones/')
    for i in etfs:
        df = yf.download(i, period = '5y', interval = '1d')
        df.to_csv(i + '.csv')



def merge(acciones):
    data = pd.read_csv(direccion + acciones[1], usecols = ['Date'])
    data.Date = pd.to_datetime(data.Date, dayfirst = True)
    time = set(data.Date)
    for i in acciones:
        data = pd.read_csv(direccion + i, usecols = ['Date'])
        data.Date = pd.to_datetime(data.Date, dayfirst = True)
        time = set(data.Date) & time
    return time

def get_data(acciones):
    df = pd.DataFrame()
    df2 = pd.DataFrame()
    time = merge(acciones)
    for i in acciones:
        data = pd.read_csv(direccion + i, usecols = ['Date', 'Close'])        
        data.Date = pd.to_datetime(data.Date, dayfirst = True)
        data = data[data.Date.isin(time)]
        data = data.reset_index()
        data['returns'] = data.Close/data.Close.shift() - 1
        df[i.replace('.csv','').replace('^','')] = data.returns
        df2[i.replace('.csv','').replace('^','')] = data.Close
        
    os.chdir('/home/cristiandavid/Documents/unsa_2022/economia/tesis_2/Optimum_portfolio_py')
    df['date'] = data.Date
    df2['date'] = data.Date
    df.dropna(axis = 0).to_csv('Retornos.csv')
    df2.dropna(axis = 0).to_csv('Precios.csv')
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

#%% Clase CAPM


class CAPM():
    '''
    Inputs:
        asset: NOmbre del activo financiero (valor de las y)
        benchmark: NOmbre del activo de mercado (valor de x)
        Ambos deben de ser Objetos de tipo String.    
    '''
    def __init__(self, asset, benchmark):
        direccion = '/home/cristiandavid/Documents/unsa_2022/economia/tesis_2/Optimum_portfolio_py/'
        self.retornos = pd.read_csv(direccion + 'Retornos.csv')
        self.precios = pd.read_csv(direccion + 'Precios.csv')
        
        self.asset = self.retornos[asset].name
        self.benchmark = self.retornos[benchmark].name
        self.y = self.retornos[asset].values
        self.x = self.retornos[benchmark].values
        self.beta = None
        self.alpha = None
        self.r_value = None
        self.p_value = None
        self.std_error = None
        
    def __str__(self):
        str_self = 'Linear Regression | asset: ' + self.asset + '| Market Asset: ' + self.benchmark + '\n' + '| Alpha (Intercept): ' + str(self.alpha) + '| Beta (Slope): ' + str(self.beta) + '\n'\
            + '| P_value: ' + str(self.p_value) + '| Null Hypthesis: ' + str(self.hypotesis) + '\n' + '|r-value: ' + str(self.r_value) + '| r-squared: ' + str(self.r_squared)
        return str_self
      

    def compute(self):
        decimals = 4
        slope, intercept, r_value, p_value, std_error = scipy.stats.linregress(self.x, self.y)
        self.beta = np.round(slope, decimals)
        self.alpha = np.round(intercept, decimals)
        self.r_value = np.round(r_value, decimals)
        self.p_value = np.round(p_value, decimals)
        self.hypotesis = self.p_value > .05 # Para valores significativos 
        self.r_squared = np.round(r_value ** 2, decimals)
        self.predictor = self.alpha + self.beta* self.x
    
    def scatter(self):
        plt.figure()
        plt.title(self.__str__())
        plt.scatter(self.x, self.y)
        plt.plot(self.x, self.predictor, c = 'r')
        plt.ylabel(self.asset)
        plt.xlabel(self.benchmark)
        # plt.grid()
        plt.tight_layout()
        plt.show()
        
    def dual_graph_normaliced(self):
        price_asset = self.precios[self.asset]
        price_benchamark = self.precios[self.benchmark]
        plt.figure()
        plt.title('Serie de tiempo de los precios de |n' + self.asset + self.benchmark + '|n Normalizados a 100')
        plt.xlabel('Tiempo')
        plt.ylabel('Precios Normalizados')
        price_asset = 100 * price_asset / price_asset[0]
        price_benchamark = 100 * price_benchamark / price_benchamark[0]
        plt.plot(price_asset, color = 'r', label = self.asset)
        plt.plot(price_benchamark, color = 'black', label = self.benchmark)
        plt.legend(loc = 0)
        plt.tight_layout()
        plt.show()
        
        
        
        
        
        
        
#%%
# primero vere mis raw materials
os.chdir('/home/cristiandavid/Documents/unsa_2022/economia/tesis_2/Optimum_portfolio_py/acciones/')
acciones = os.listdir()
# En accioens se encuentra una lista con todos los activos descargados en formato
# csv
# Posteriomente procesare cada uno de estos datos y los unire en un solo df
get_data(acciones)
#%%
# Probando los algoritmos..
df.columns
market_assets = ['STOXX50E','STOXX','S&P500']


capm = CAPM('SAN.MC', 'STOXX')
capm.compute()
capm.scatter()
capm.alpha
capm.dual_graph_normaliced()


# Creando la base de datos, tambien conocieda como la matriz de precios y retornos
# price_matrix, returns_matrix = base_retornos(acciones)
df = pd.read_csv('Precios.csv')
df.index = df.date

# Ahora veamos el enfoque del modelo CAPM
# Para lo cual necesitamos hacer una regresion lineal teniendo como variable 
# dependiente al vector de retornos de una accion y como independiente al vector 
# retornos de un activo de mercado. Como seria el caso de un indice de mercado
# Definamos el vector de retornos del activo de mercado
# Dado que tenemos acciones de mercados europeros tomaremos en cuenta al STOXX Y 
# STOXX50
#%%DESCARGAR DE LOS DATOS A USAR 
# Definire una lista con el nombre de todos los activos que usare 
# La informacion presentada saldra de la lista de SPDR etf listing 
# en la seccion de INDUSTRY 
etfs = ['KBE','KRE','KCE','KIE','XAR','XTN','XBI','XPH','XHE','XHS','XOP','XME','XRT','XHB','XSD','XSW','XNTK','XITK','XTL','XWEB','EPU']
# Para poder descargarlos usare la siguientie lireria de pandas 
yahoo_api('S&P500')














































