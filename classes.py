# Las clases son un archivo
class jb():
    
    def __init__(self, x):
    
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
