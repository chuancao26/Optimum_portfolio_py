# Apartado de modulos en python+++
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import importlib
from scipy.stats import skew, kurtosis, chi2

import os 
direccion = '/media/cristian/88B2BE6CB2BE5DFE/2022_UNSA/semestre par/economia/tesis_2/PORTAFOLIO_PY/Optimum_portfolio_py/'
os.chdir(direccion)
import functions
importlib.reload(functions)
import classes
importlib.reload(classes)
#%%
# vamos a hacer funciones para poder reutilzar todo este codigo
acciones = os.listdir(direccion + '/acciones/')
acciones
# ['DBK.DE.csv', '^IXIC.csv']
#%%
# veamos los inputs
ric = '^TYX' # Out[12]: ['GE.csv', 'MCD.csv', 'META.csv', '^IXIC.csv', '^TYX.csv']
path = direccion + '/acciones/'
file_extension = 'csv'
x, x_str, t = functions.load_time_series(ric, file_extension)
# Ahora voy a iniciar el constructor de mi clase
jb = classes.jb(x, x_str)
jb.compute()
jb.mean
dir(jb)
jb.str_stats()
print(jb)
functions.time_series_graph_of(t, ric)
functions.distributions_graph_of(x, ric, jb.str_stats())






