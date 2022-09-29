import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import importlib
from scipy.stats import skew, kurtosis, chi2

import os 

os.chdir('E:/UNSA 2022 CC/semestre par/tesis/acciones')
#%%
# generate random variables
x_size = 10**6
degrees_freedom = 5000
type_random_variables = 'normal' #['student','normal','exponential','chi-square]
title = 'Histograma ' + type_random_variables



if type_random_variables == 'normal':
    x = np.random.standard_normal(x_size)
elif type_random_variables == 'exponential':   
    x = np.random.standard_exponential(x_size)
elif type_random_variables == 'student':
    x = np.random.standard_t(size = x_size, df = degrees_freedom)
    title = 'Histograma ' + type_random_variables + '\nDf = ' + str(degrees_freedom)
elif type_random_variables == 'chisquared':
    x = np.random.chisquare(size = x_size, df = 2)
    title = 'Histograma ' + type_random_variables + '\nDf = ' + str(degrees_freedom)


# graficando
plt.hist(x, 
         bins = 100,
         color = 'r')
plt.title(title)
# plt.xlabel()
plt.show()

# hallando algunas metricas
x_mean = np.mean(x)
x_std = np.std(x)
x_skew = skew(x)
x_kurt = kurtosis(x) #Esta es medida teniendo como base 3 y olo que muestra es 
# el exceso
# calculando percentiles
x_median = np.median(x)
x_var_95 = np.percentile(x, 5)
# Hallando el Jarque bera
x_JB = x_size/6*(x_skew**2+1/4*x_kurt**2)
# Hallando P value
# como sabemos que el jb se distribuye como una chi de 2 df
# podemos usar esta distribucion cumulativa para hallar los valores 
# fuera del limite
p_value = 1 - chi2.cdf(x_JB, df = 2)
is_normal = (p_value > .05)


# efectivamente
# Hallando el valor en riesgo al95
# print en consola
print('Distribucion ' + type_random_variables)
print('media: ', x_mean)
print('std: ', x_std)
print('skew: ', x_skew)
print('kurtosis: ', x_kurt)
print('Var 95%: ', x_var_95)
print('Jarque bera: ',x_JB)
print('P_value: ', p_value)
print('is normal: ', is_normal)


# Como queremos hallar el estadistico de jarque bera, necesitamos 
# el skew y la kurtosis
 
#%% Probando la hipotesis 
x_size = 10**6
degrees_freedom = 5000

counter = 0
while is_normal:
    type_random_variables = 'normal' #['student','normal','exponential','chi-square]
    # title = 'Histograma ' + type_random_variables 
    
    if type_random_variables == 'normal':
        x = np.random.standard_normal(x_size)
    elif type_random_variables == 'exponential':   
        x = np.random.standard_exponential(x_size)
    elif type_random_variables == 'student':
        x = np.random.standard_t(size = x_size, df = degrees_freedom)
        # title = 'Histograma ' + type_random_variables + '\nDf = ' + str(degrees_freedom)
    elif type_random_variables == 'chisquared':
        x = np.random.chisquare(size = x_size, df = 2)
        # title = 'Histograma ' + type_random_variables + '\nDf = ' + str(degrees_freedom)
    
    # # graficando
    # plt.hist(x, 
    #          bins = 100,
    #          color = 'r')
    # plt.title(title)
    # # plt.xlabel()
    # plt.show()
    
    # hallando algunas metricas
    x_mean = np.mean(x)
    x_std = np.std(x)
    x_skew = skew(x)
    x_kurt = kurtosis(x) #Esta es medida teniendo como base 3 y olo que muestra es 
    # el exceso
    # calculando percentiles
    x_median = np.median(x)
    x_var_95 = np.percentile(x, 5)
    # Hallando el Jarque bera
    x_JB = x_size/6*(x_skew**2+1/4*x_kurt**2)
    # Hallando P value
    # como sabemos que el jb se distribuye como una chi de 2 df
    # podemos usar esta distribucion cumulativa para hallar los valores 
    # fuera del limite
    p_value = 1 - chi2.cdf(x_JB, df = 2)
    is_normal = (p_value > .05)
    
    
    # efectivamente
    # Hallando el valor en riesgo al95
    # print en consola
    print('Distribucion ' + type_random_variables)
    print('media: ', x_mean)
    print('std: ', x_std)
    print('skew: ', x_skew)
    print('kurtosis: ', x_kurt)
    print('Var 95%: ', x_var_95)
    print('Jarque bera: ',x_JB)
    print('P_value: ', p_value)
    print('is normal: ', is_normal)
    print('counter = ', counter)
    counter += 1
    # Como queremos hallar el estadistico de jarque bera, necesitamos 
    # el skew y la kurtosis

#%% VIENDO DATOS DE MERCADOS REALES 

os.chdir('E:/UNSA 2022 CC/semestre par/tesis/acciones')

# leyendo los 
dir(os)
help(os.listdir)
acciones = os.listdir('E:/UNSA 2022 CC/semestre par/tesis/acciones')
# ['DBK.DE.csv', '^IXIC.csv']
#%%
# lista con los elemetos totales
# Out[17]: ['GE.csv', 'MCD.csv', 'META.csv', '^IXIC.csv', '^TYX.csv']
acciones
ric = acciones[-1]
path = 'E:/UNSA 2022 CC/semestre par/tesis/acciones/'
# haciendo movil la entrada de datos
table_raw = pd.read_csv(path + ric)
table_raw.shape
table_raw.columns
table_raw.Date
# ahora convertiremos estos dates simples en datetimes
# para lo q creare un df
t = pd.DataFrame()
t['Date'] = pd.to_datetime(table_raw['Date'], dayfirst = True)
t['close'] = table_raw['Close']
# Hallando los retornos 
t['close_previous'] = table_raw.Close.shift(1)
t['return_close'] = t.close/t.close_previous - 1

# veamos una grafica de time series
plt.figure()
plt.plot(t.Date, t.close)
plt.title('Time series de ' + ric[:ric.rfind('.csv')])
plt.xlabel('Time')
plt.ylabel('Price') 
plt.show()
# help(plt.plot)

# reciclano el codigo para las metricas estadisticas
# Eliminando los na
t = t.dropna()
x = t.return_close

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
# Hallando P value
# como sabemos que el jb se distribuye como una chi de 2 df
# podemos usar esta distribucion cumulativa para hallar los valores 
# fuera del limite
p_value = 1 - chi2.cdf(x_JB, df = 2)
is_normal = (p_value > .05)


# efectivamente
# Hallando el valor en riesgo al95
# print en consola
print('Analasis para ' + ric)
print('media: ', x_mean)
print('std: ', x_std)
print('skew: ', x_skew)
print('kurtosis: ', x_kurt)
print('Var 95%: ', x_var_95)
print('C_var_95%: ', x_Cvar_95)
print('Jarque bera: ',x_JB)
print('P_value: ', p_value)
print('is normal: ', is_normal)


# Graficando la distribucion de los retornos
plt.figure()
plt.hist(x, bins = 100)
plt.title('Grafico de distribucion para la variable ' + ric[:ric.rfind('.csv')])
plt.show()
dir(acciones[1])




