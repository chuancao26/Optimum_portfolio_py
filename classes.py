# Las clases son un archivo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import importlib
from scipy.stats import skew, kurtosis, chi2

import os 

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
        
        
        
        
        
        
        
        
        
        
        
        
        