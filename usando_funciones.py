# Apartado de modulos en python+++
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import importlib
from scipy.stats import skew, kurtosis, chi2

import functions
importlib.reload(functions)


import os 

os.chdir('E:/UNSA 2022 CC/semestre par/tesis')

#%%
# vamos a hacer funciones para poder reutilzar todo este codigo
os.chdir('E:/UNSA 2022 CC/semestre par/tesis/acciones')
acciones = os.listdir('E:/UNSA 2022 CC/semestre par/tesis/acciones')
# ['DBK.DE.csv', '^IXIC.csv']
#%%
# veamos los inputs
acciones
ric = '^TYX' # Out[12]: ['GE.csv', 'MCD.csv', 'META.csv', '^IXIC.csv', '^TYX.csv']
path = 'E:/UNSA 2022 CC/semestre par/tesis/acciones/'
file_extension = 'csv'
x, x_str, t = functions.load_time_series(ric, file_extension)
# haciendo movil la entrada de dato
x_size = len(x)

x_mean = np.mean(x)
x_std = np.std(x)
x_skew = skew(x)
x_kurt = kurtosis(x) #Esta es medida teniendo como base 3 y olo que muestra es 
# el exceso
# calculando percentiles
x_median = np.median(x)
x_var_95 = np.percentile(x, 5)
# Hallando el Jarque bera
x_Cvar_95 = np.mean(x[x <= np.percentile(x, 5)])
x_JB = x_size/6*(x_skew**2+1/4*x_kurt**2)
x_sharpe = x_mean/x_std * np.sqrt(252)
# Hallando P value
# como sabemos que el jb se distribuye como una chi de 2 df
# podemos usar esta distribucion cumulativa para hallar los valores 
# fuera del limite
p_value = 1 - chi2.cdf(x_JB, df = 2)
is_normal = (p_value > .05)


# Time series graph of the stock
# plt.figure()
# plt.plot(t.Date, t.close, color = 'r')
# plt.title('Time Series Graph of ' + ric)
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.show()

# efectivamente
# Hallando el valor en riesgo al95
# print en consola
print('Analasis para ' + ric)
print('media: ', x_mean)
print('std: ', x_std)
print('skew: ', x_skew)
print('kurtosis: ', x_kurt)
print('Var 95%: ', x_var_95)
print('Sharpe: ', x_sharpe)
print('C_var_95%: ', x_Cvar_95)
print('Jarque bera: ',x_JB)
print('P_value: ', p_value)
print('is normal: ', is_normal)


# Colocando los xlabels que requerimos 
digits = 3
string_plot = 'Analisis para el activo: ' + ric + '\n' + ' | Media: ' + str(np.round(x_mean, digits)) \
     + '| stdeviation: ' + str(np.round(x_std,digits)) + '| Skewness: ' + str(np.round(x_skew , digits)) \
     + '| Kurtosis: ' + str(np.round(x_kurt, digits)) + '| Var 95: ' + str(np.round(x_var_95, digits)) + '\n' \
     + 'Sharpe: ' + str(np.round(x_sharpe, digits)) + '| CVAR 95: ' + str(np.round(x_Cvar_95, digits)) + '| JB: ' + str(np.round(x_JB, digits)) \
    + '| p_value: ' + str(np.round(p_value, digits)) + '| is_normal: ' + str(is_normal)



functions.time_series_graph_of(t, ric)
functions.distributions_graph_of(x, ric, string_plot)
#%%
# Graficando la distribucion de los retornos
plt.figure()
plt.hist(x, bins = 100)
plt.title('Grafico de distribucion para la variable ' + ric[:ric.rfind('.csv')])
plt.xlabel(str1 + '\n' + str2)
plt.show()
dir(acciones[1])








