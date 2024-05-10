# Apartado de librerias necesarias para el procesamiento de datos 
import os 
import classes 
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import importlib
from scipy.stats import skew, kurtosis, chi2
import yfinance as yf
import cvxpy as cv #libreria de optimizacion
importlib.reload(classes)
# Ahora importo los scripts necesarios, como classes y functions para cargar los 
# datos

#%% Funciones 
direccion = 'acciones/'
direccion2 = 'Optimum_portfolio_py/'
def yahoo_api(etfs):
    """
        Inputs:
            iterable, con el nombre de los activos   
        Guarda los archivos .csv en la carpeta acciones
    """
    ruta_carpeta = 'acciones'
    # Verificar si la carpeta ya existe
    if os.path.exists(ruta_carpeta):
        # Si existe, eliminarla
        shutil.rmtree(ruta_carpeta)
        print("La carpeta existente ha sido eliminada.")
    
    # Crear la carpeta
    os.makedirs(ruta_carpeta)
    print("La carpeta se ha creado exitosamente.")
    for i in etfs:
        df = yf.download(i, period='5y', interval='1d')
        ruta_archivo = os.path.join(ruta_carpeta, i + '.csv')  # Construye la ruta completa del archivo
        print(ruta_archivo)
        df.to_csv(ruta_archivo)
        print(f"Archivo '{i}.csv' guardado en la carpeta 'acciones'.")




def merge(acciones, direccion):
    # Lee el primer archivo para obtener las fechas
    data = pd.read_csv(direccion + acciones[0], usecols=['Date'])
    data.Date = pd.to_datetime(data.Date, dayfirst=True)
    time = set(data.Date)
    
    # Itera sobre el resto de archivos para encontrar las fechas comunes
    for i in acciones[1:]:
        data = pd.read_csv(direccion + i, usecols=['Date'])
        data.Date = pd.to_datetime(data.Date, dayfirst=True)
        time = set(data.Date) & time
    
    return time

def get_data(acciones):
    """
    Parameters
    ----------
    acciones : iterable (list)
        La lista de acciones tiene que contener los nombres de cada uno de los 
        activos en formato csv
    direccion : str
        La ruta donde se encuentran los archivos csv
    Returns
    -------
    La funcion guarda en disco duro 2 archivos en formato csv los cuales son 
    el DataFrame de los precios y de los retornos

    """
    # Obtiene las fechas comunes
    time = merge(acciones, direccion)
    
    # Define los DataFrames para los precios y los retornos
    df = pd.DataFrame()
    df2 = pd.DataFrame()
    
    # Itera sobre las acciones y crea los DataFrames
    for i in acciones:
        data = pd.read_csv('acciones/' + i, usecols=['Date', 'Close'])        
        data.Date = pd.to_datetime(data.Date, dayfirst=True)
        data = data[data.Date.isin(time)]
        data = data.reset_index()
        data['returns'] = data.Close / data.Close.shift() - 1
        df[i.replace('.csv', '').replace('^', '')] = data.returns
        df2[i.replace('.csv', '').replace('^', '')] = data.Close
        
    # Añade la columna de fecha a los DataFrames
    df['date'] = data.Date
    df2['date'] = data.Date
    
    # Guarda los archivos CSV
    df.dropna(axis=0).to_csv('Retornos.csv', index=False)
    df2.dropna(axis=0).to_csv('Precios.csv', index=False)
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
        direccion = 'C:/Users/quiroz/Documents/Unsa/economia/tesis/Optimum_portfolio_py/'
        self.retornos = pd.read_csv(direccion + 'Retornos.csv')
        self.precios = pd.read_csv(direccion + 'Precios.csv')
        self.precios.date = pd.to_datetime(self.precios.date, dayfirst = True)
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
        time = self.precios.date
        price_asset = self.precios[self.asset]
        price_benchamark = self.precios[self.benchmark]
        plt.figure()
        plt.title('Serie de tiempo de los precios de\n' + self.asset + ' y ' + self.benchmark + ' como activo de mercado' + '\n Normalizados a 100')
        plt.xlabel('Tiempo')
        plt.ylabel('Precios Normalizados')
        price_asset = 100 * price_asset / price_asset[0]
        price_benchamark = 100 * price_benchamark / price_benchamark[0]
        plt.plot(time, price_asset, color = 'r', label = self.asset)
        plt.plot(time, price_benchamark, color = 'black', label = self.benchmark)
        plt.legend(loc = 0)
        plt.tight_layout()
        plt.show()
    def dual_axes(self):
        
        plt.figure(figsize = (12,5))
        plt.title('Grafico de Series de tiempo dual para \n' + self.asset +  ' y el activo de mercado ' + self.benchmark )
        plt.xlabel('Tiempo')
        plt.ylabel('Precios')
        ax = plt.gca()
        ax1 = self.precios.plot(kind = 'line', x = 'date', y = self.asset, ax = ax, c = 'blue',label = self.asset)
        ax2 = self.precios.plot(kind = 'line', x = 'date', y = self.benchmark, ax = ax, c = 'red',label = self.benchmark, secondary_y = True)
        ax1.legend(loc = 2)
        ax2.legend(loc = 1)
        plt.tight_layout()
        plt.show()
#%% Analisis estadistico
class jb():
    
    def __init__(self, x, x_str_name):
        self.name = x_str_name
        self.digits = 4
        self.returns = x
        self.size = len(x)
        self.mean = 0
        self.std = 0
        self.skew = 0
        self.kurt = 0 
        self.median = 0
        self.var_95 = 0
        # Hallando el Jarque bera
        self.Cvar_95 = 0
        self.JB = 0
        self.sharpe = 0
        # Hallando P value
        # como sabemos que el jb se distribuye como una chi de 2 df
        # podemos usar esta distribucion cumulativa para hallar los valores 
        # fuera del limite
        self.p_value = 0
        self.is_normal = False
    def __str__(self):
        to_print = self.name + '|' + str(self.size)+ '\n' + str(self.str_stats())
        return to_print
        

    def compute(self):
        self.mean = np.mean(self.returns)
        self.std = np.std(self.returns)
        self.skew = skew(self.returns)
        self.kurt = kurtosis(self.returns) #Esta es medida teniendo como base 3 y olo que muestra es 
        # el exceso
        # calculando percentiles
        self.median = np.median(self.returns)
        self.var_95 = np.percentile(self.returns, 5)
        # Hallando el Jarque bera
        self.Cvar_95 = np.mean(self.returns[self.returns <= np.percentile(self.returns, 5)])
        self.JB = self.size/6*(self.skew**2+1/4*self.kurt**2)
        self.sharpe = self.mean/self.std * np.sqrt(252)
        # Hallando P value
        # como sabemos que el jb se distribuye como una chi de 2 df
        # podemos usar esta distribucion cumulativa para hallar los valores 
        # fuera del limite
        self.p_value = 1 - chi2.cdf(self.JB, df = 2)
        self.is_normal = (self.p_value > .05)
        
    def str_stats(self):
        str_stats ='| Media: ' + str(np.round(self.mean, self.digits)) \
             + '| stdeviation: ' + str(np.round(self.std,self.digits)) + '| Skewness: ' + str(np.round(self.skew , self.digits)) \
             + '| Kurtosis: ' + str(np.round(self.kurt, self.digits)) + '| Var 95: ' + str(np.round(self.var_95, self.digits)) + '\n' \
             + 'Sharpe: ' + str(np.round(self.sharpe, self.digits)) + '| CVAR 95: ' + str(np.round(self.Cvar_95, self.digits)) + '| JB: ' + str(np.round(self.JB, self.digits)) \
            + '| p_value: ' + str(np.round(self.p_value, self.digits)) + '| is_normal: ' + str(self.is_normal)
        return str_stats
    
    def resume():
        fig, ax = plt.subplots(figsize = (16,9), dpi = 200)
        x = np.array(tabla3.Riesgo)
        y = np.array(tabla3.Media)

        ax.scatter(x,
                   y,
                   color = 'green',
                   s = 50)
        ax.set_ylabel('Retorno',
                      fontfamily = 'serif',
                      fontweight = 'regular',
                      fontsize = 12,
                      loc = 'center')
        ax.set_xlabel('Riesgo',
                      fontfamily = 'serif',
                      fontweight = 'regular',
                      fontsize = 12,
                      loc = 'center')
        
 
        
#%% Descargar de los precios de los activos (ETFs)
# Definire una lista con el nombre de todos los activos que usare 
# La informacion presentada saldra de la lista de SPDR etf listing 
# ETFs americanos: Industry (Modified Equal Weighted)
# Fuente: https://www.ssga.com/library-content/products/fund-docs/etfs/us/information-schedules/spdr-etf-listing.pdf
# ETF peruano: EPU
# Fuente: https://www.blackrock.com/cl/productos/239606/ishares-msci-all-peru-capped-etf
# El activo de mercado sera el S&P500 para el mercado estadounidense: ^GSPC
# El activo de mercado para el mercado peruano sera: S&P Lima General (SPBLPGPT)

etfs = ['KBE','KRE','KCE','KIE','XAR','XTN','XBI','XPH','XHE','XHS','XOP','XME','XRT','XHB','XSD','XSW','XNTK','XITK','XTL','XWEB','EPU','^GSPC']

# Para poder descargarlos usare la siguientie lireria de pandas: yfinance
# Los datos seran descargados seran de 5 años hasta el 5 de noviembre del 2022.
# la frecuencia sera diaria 
# Usando la funcion:
    
    
yahoo_api(etfs)


# En total son 21 ETFs , 1 activo de mercado y el treasury 10 years 
# len(etfs) #22

# del etfs
#%%
# Una vez descargados todos los archivos necesarios, procedo a procesarlos.
# Abro la carpeta contenedora de los archivos que estan separados y en formato .csv
# Asiganare todos los elementos en la variable acciones
acciones = os.listdir('acciones')[:-1]
# ['EPU.csv',
#  'KBE.csv',
#  'KCE.csv',
#  'KIE.csv',
#  'KRE.csv',
#  'XAR.csv',
#  'XBI.csv',
#  'XHB.csv',
#  'XHE.csv',
#  'XHS.csv',
#  'XITK.csv',
#  'XME.csv',
#  'XNTK.csv',
#  'XOP.csv',
#  'XPH.csv',
#  'XRT.csv',
#  'XSD.csv',
#  'XSW.csv',
#  'XTL.csv',
#  'XTN.csv',
#  'XWEB.csv',
#  '^GSPC.csv',
#  '^TNX.csv'] 
        
# Con esto, procedo a unir estos datos en una sola base de datos, o tambien llamado
# DataFrame usando la funcion que programe get_data()

get_data(acciones)

acciones
#%%
# Probando los algoritmos..

capm = CAPM('KBE','GSPC')
capm.compute()
capm.scatter()
capm.alpha
capm.dual_graph_normaliced()
capm.dual_axes()



#%%
# Indicadores:
# Media de los retornos de cada etf
# cargando las base de datos de precios y retornos de los activos 
os.chdir('C:/Users/quiroz/Documents/Unsa/economia/tesis/Optimum_portfolio_py/')
precios = pd.read_csv('Precios.csv')
retornos = pd.read_csv('Retornos.csv')

# cambiando los dates a datetime
precios.date = pd.to_datetime(precios.date, dayfirst = True)
retornos.date = pd.to_datetime(retornos.date, dayfirst = True)

precios.columns
# Borrando la columna unnamed por ser innecesaria
precios = precios.drop('Unnamed: 0', axis = 1)
retornos = retornos.drop('Unnamed: 0', axis = 1)

#%%
objeto = jb(retornos.EPU, 'EPU')
objeto.compute()

#%% Calculando la matriz de varianza covarianza

def covarianza_correlacion(df, scale = False, decimals = 4):
    """
    Parameters
    ----------
    df : dataframe de retornos (solo retornos)
    scale: bool type, factor 252
    decimals = 4, cantidad de decimales para redondeo
    
    Returns    
    -------
    1. Matriz de covarianzas (Array_like)
    2. Matriz de correlacion (Array_like)
    """
    largo = len(retornos.columns)
    matriz = np.zeros([largo,largo])
    correlacion = np.zeros([largo,largo])
    for pos, i in enumerate(df.columns):
        for posj, j  in enumerate(df.columns):
            if scale:
                tmp_cov = np.cov(df[i], df[j])[0][1] * 252
                tmp_corr = np.corrcoef(df[i], df[j])[0][1]
                matriz[pos][posj] = np.round(tmp_cov, decimals)
                correlacion[pos][posj] = np.round(tmp_corr, decimals)
            else:
                tmp_cov = np.cov(df[i], df[j])[0][1]
                tmp_corr = np.corrcoef(df[i], df[j])[0][1]
                matriz[pos][posj] = np.round(tmp_cov, decimals)
                correlacion[pos][posj] = np.round(tmp_corr, decimals)
    return matriz, correlacion


#%%
# Cambiando los nombres de las columnas 
new_names = {'KBE':'BANKS',
             'KRE':'REGIONAL_BANKS',
             'KCE':'CAPITAL_MARKETS',
             'KIE':'INSURANCE',
             'XAR':'AEROESPACIAL_DEFENSE',
             'XTN':'TRANSPORT',
             'XBI':'BIOTECH',
             'XPH':'PHARMACEUTICALS',
             'XHE':'HEALTH_CARE_EQUIPMENT',
             'XHS':'HEALTH_CARE_SERVICES',
             'XOP':'GAS_EXPLORATION',
             'XES':'GAS_EQUIPMENT',
             'XHB':'HOMEBUILDERS',
             'XME':'METALS_MINING',
             'XITK':'FACTSET_TECHNOLOGY',
             'XNTK':'NYSE_TECHNOLOGY',
             'XRT':'RETAIL',
             'XSD':'SEMICONDUCTOR',
             'XSW':'SOFTWARE',
             'XTL':'TELECOM',
             'XWEB':'INTERNET',
             'EPU':'PERUVIAN_FIRMS',
             'GSPC':'S&P500'}


retornos_name = retornos.copy()
retornos_name = retornos_name.rename(columns = new_names)
retornos_name = retornos_name.drop(['date'], axis = 1)
retornos = retornos.drop(['date'], axis = 1)


#%% TABLAS
# Hallar la tabla 2:
# Obteniendo los nombres
tabla1 = pd.Series(new_names).reset_index()
tabla1.columns = ['indice','nombre']
tabla1.nombre = tabla1.nombre.str.capitalize()
tabla1
tabla1.to_excel("tabla_names.xlsx")
#%%
# Hallando la tabla 3 Media y riesgo

tabla3 = retornos.apply(['mean', 'var']).T
tabla3 
tabla3['var'] = (tabla3['var'] * 252) ** .5
tabla3['mean'] = (1 + tabla3['mean']) ** 252 -1
tabla3.columns = ['Media', 'Riesgo']

codes = tabla3.sort_values(by = 'Media', ascending = False).index

# //volviendo minusculas y ordenando
tabla3.index = retornos_name.columns.str.capitalize()
tabla3 = tabla3.sort_values(by = 'Media', ascending = False)
tabla3 
help(tabla3.index.str.lower)
# Hallando el sharpe
yahoo_api(['^TNX'])
os.chdir(direccion)
riskfree = pd.read_csv('^TNX.csv')
riskfree = riskfree.Close/100
riskfree.name = 'Risk Free'

tabla3r = tabla3.append({'Media' : riskfree.mean(), 'Riesgo' : riskfree.std()}, ignore_index = True)
indice3 = list(tabla3.index)
indice3.append('RiskFree')
tabla3r.index = indice3
tabla3r['Ratio de Sharpe'] = (tabla3r.Media - riskfree.mean()) / tabla3r.Riesgo
tabla3r
os.chdir(direccion2)
tabla3r.to_excel('tabla3.xlsx')

#%% Tabla 4
# Matriz de covarianzas y Varianzas
mat_cov = retornos.cov() * 252
mat_cov.to_excel('tabla4.xlsx')
# Matriz de correlaciones

mat_corr = np.round(retornos.corr(), 2)
mat_corr.to_excel('tabla5.xlsx')


#%% GRAFICOS

fig, ax = plt.subplots(figsize = (16,9), dpi = 200)
x = np.array(tabla3.Riesgo)
y = np.array(tabla3.Media)

ax.scatter(x,
           y,
           color = 'green',
           s = 50)
ax.set_ylabel('Retorno',
              fontfamily = 'serif',
              fontweight = 'regular',
              fontsize = 12,
              loc = 'center')
ax.set_xlabel('Riesgo',
              fontfamily = 'serif',
              fontweight = 'regular',
              fontsize = 12,
              loc = 'center')
for n, i in enumerate(codes):
    ax.annotate(i, (x[n], y[n]), verticalalignment='bottom',
                horizontalalignment = 'center')
    

plt.tight_layout()
plt.savefig('every_retornos_riesgo.png')
plt.show()

#%% Segmentando
def sharpe(x, rf = np.mean(riskfree)):
    return (((1 + np.mean(x)) ** 252 - 1 - rf) / (np.var(x)*252)**.5) 
# Hallando el ratio de sharpe para cada activo para pode elegir a los mas optimos
seg = sharpe(retornos.drop('GSPC', axis = 1))
# elegire a los 10 mas optimos
seg_10 = seg.sort_values(ascending = False)[:9].index.values
seg_10 = np.append(seg_10, 'EPU')

#%%
# Optimizacion:
# Frontera eficiente:
# tenemos en total 22 activos 
retornos_opti = retornos.copy()

# Seleccionando los activos a tomar en cuenta.
retornos_opti = retornos_opti[seg_10]
import time 
len(retornos_opti.columns)
# Para optimizar y crear la cartera optima dadas las indicacioes 
# Usaremos el poder de calculo del computaor para hacer 100000 iteraciones para 
# crear distintas portafolios que nos ayudaran a formar la frontera eficiente 
inicio = time.time()
iteraciones = 100000
cov = retornos_opti.cov() * 252
cols = list(retornos_opti.columns)
cols = cols + ['Retorno','Riesgo']
cols
retornos_mean = (1 + retornos_opti.mean()) ** 252 - 1
orden = retornos_mean.index
retornos_mean = np.array(retornos_mean) 
tmp = pd.DataFrame()
for i in range(iteraciones):
    port_weights = np.random.uniform(0,1,len(retornos_opti.columns))
    port_weights = port_weights / port_weights.sum()
    port_retorno = (port_weights  * retornos_mean).sum()
    port_riesgo = ((np.array(port_weights).dot(np.array(cov))).dot(np.transpose(np.array(port_weights)))) ** 0.5
    tmp_array = np.append(port_weights, port_retorno)
    tmp_array = np.append(tmp_array, port_riesgo)
    tmp[str(i)] = tmp_array
frontera = tmp.T
frontera.columns = cols
frontera.Retorno.plot(kind = 'hist')
frontera.Riesgo.plot(kind = 'hist')
fin = time.time()
print('Tiempo empleado: ', fin - inicio)

#%%
frontera.Retorno.max()
fig, ax = plt.subplots(figsize = (16,9), dpi = 200)
x = frontera.Riesgo
y = frontera.Retorno

ax.scatter(x,
           y,
           color = 'green',
           s = 50)
ax.set_ylabel('Retorno de Portafolio',
              fontfamily = 'serif',
              fontweight = 'regular',
              fontsize = 12,
              loc = 'center')
ax.set_xlabel('Riesgo de portafolio',
              fontfamily = 'serif',
              fontweight = 'regular',
              fontsize = 12,
              loc = 'center')
plt.tight_layout()
plt.savefig('frontera.png')
plt.show()




#%%
# =============================================================================
# CALCULANDO LOS PESOS PARA CADA TIPO DE PORTAFOLIO ESPECIFICO
# =============================================================================

 # Portafolio Equiponderado
cov = retornos_opti.cov() * 252
retornos_mean = (1 + retornos_opti.mean()) ** 252 - 1 
retornos_mean 
weightsEqui= np.ones([1,len(seg_10)]) * 1/len(seg_10)


#%%
#creando los inputs del modelo de media varianza
retornos_opti = retornos[seg_10]

mu = np.array(np.mean(retornos_opti, axis = 0),ndmin = 2)
sigma = np.cov(retornos_opti.values.T)
w = cv.Variable(mu.shape[1])
g1 = cv.Parameter(nonneg=True)
g1.value = 1
ret = mu @ w
risk = cv.quad_form(w, sigma)
risk.shape
#definiendo la funcion objetivo
prob = cv.Problem(cv.Maximize(ret - g1*risk),
#añadiendo las restriciones, aca se puede añadir las restricciones que deseemos sobre los pesos,
#otros indicadores como MAD, CVaR, entre otros dependiendo de como se modele el problema
                   [cv.sum(w) == 1,
                    w >= 0.01])#esta restriccion para que cada activo tenga como minimo 1%
#resolviendo el problema
prob.solve()
weightsMax = np.array(w.value, ndmin=2)


#%%
####################################################################
# Modelo Sharpe Ratio Media Varianza
####################################################################
retornos_opti = retornos[seg_10]
#definiendo los inputs
mu = np.array(np.mean(retornos_opti, axis = 0),ndmin = 2)
sigma = np.cov(retornos_opti.values.T)
w = cv.Variable(mu.shape[1])
k = cv.Variable(1)
rf=cv.Parameter(nonneg=True)
rf.value = 0
u=np.ones((1,mu.shape[1]))*rf

#definiendo el problema, funcion objetivo y reestricciones
prob = cv.Problem(cv.Minimize(cv.quad_form(w,sigma)),
               [(mu-u) @ w == 1,
               w >= 0,
               k >= 0,
               w >= 0.01*k, #para que el peso minimo sea 0.01%
               cv.sum(w) == k])
#resolviendo el problema
prob.solve(solver=cv.ECOS)
weightsSharpe = np.array(w.value/k.value, ndmin=2)

#%%
####################################################################
# Modelo Min
####################################################################
retornos_opti = retornos[seg_10]
#definiendo los inputs
mu = np.array(np.mean(retornos_opti, axis = 0),ndmin = 2)
sigma = np.cov(retornos_opti.values.T)
w = cv.Variable(mu.shape[1])
k = cv.Variable(1)
rf=cv.Parameter(nonneg=True)
rf.value = 0
u=np.ones((1,mu.shape[1]))*rf

#definiendo el problema, funcion objetivo y reestricciones
prob = cv.Problem(cv.Minimize(cv.quad_form(w,sigma)),
              [cv.sum(w) == 1,
               w >= 0.01])
#resolviendo el problema
prob.solve(solver=cv.ECOS)
weightsMin = np.array(w.value, ndmin=2)


#%%
weights = [weightsEqui,weightsMax, weightsMin, weightsSharpe] #1,10
retornos_opti = retornos[seg_10]
cov = np.array(retornos_opti.cov())
exp_return = np.array(retornos_opti.mean()) #10,1
data = list(seg_10)
data = data + ['Retorno','Riesgo','Sharpe']
portafolios = pd.DataFrame()
cont = 0
for i in weights:
    # print(i)
    weight = i
    retorno_portafolio = (weight.dot(exp_return)) * 252
    riesgo = ((weight.dot(np.array(cov) * 252)).dot(np.transpose(weight))) ** 0.5
    sharpe = (retorno_portafolio - riskfree.mean()) / riesgo
    weight = np.append(weight, retorno_portafolio)
    weight = np.append(weight, riesgo)
    weight = np.append(weight, sharpe)
    portafolios[str(cont)] = weight
    cont += 1
portafolios = portafolios.T
portafolios.columns = data
indice = ['Equiponderado','Portafolio Máximo retorno','Portafolio Mínima Varianza',"Portafolio Maximo Sharpe"]
portafolios.index = indice
portafolios.to_excel('tabla7.xlsx')



