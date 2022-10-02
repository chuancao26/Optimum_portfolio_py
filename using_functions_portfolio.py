# Apartado de librerias necesarias para el procesamiento de datos 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import importlib
from scipy.stats import skew, kurtosis, chi2

import os 

# Ahora importo los scripts necesarios, como classes y functions para cargar los 
# datos

import functions
importlib.reload(functions)
import classes
importlib.reload(classes)

os.chdir('/media/cristiand/CURSOS FEC/UNSA 2022 CC/semestre par/ECONOMIA/tesis/Optimum_portfolio_py/')

#%%
# primero vere mis raw materials

os.chdir('/media/cristiand/CURSOS FEC/UNSA 2022 CC/semestre par/ECONOMIA/tesis/Optimum_portfolio_py/acciones')
acciones = os.listdir()

# Primero cargemos los datos 
# usando las funciones presentes en el script de funciones 
dir(functions)
ric = 'SOGN.PA'
ric2 = 'SGREN.MC'
x, x_str, t = functions.load_time_series(ric, 'csv')
x1, x_str1, t1 = functions.load_time_series(ric2, 'csv')

# lo que queremos es juntar en un dataframe los retornos de todas las acciones en 
# un dataframe
# veamos la cantidad de elementos de cada uno de los archivos en la carpeta acciones
acciones
len_acciones = {}
for i in acciones:
    print(i)
    len_acciones[i.replace('.csv', '')] = len(functions.date(i))
pd.Series(len_acciones).sort_values(ascending =False).index[-1]

len(functions.date(ric+'.csv'))
len(functions.date(ric2+'.csv'))

# veamos el proceso de filtracion para obtener un solo df con el date minimo
t1 = t1[t1.Date.isin((set(t.Date) & set(t1.Date)))]
t
t1
