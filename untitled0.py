# dada una cadena de caracteres identificar los numeros enteros
# para esto crearemos una funcion 
def isEntero(cadena):
    posicion = []
    for char in cadena: 
        try:
            int(char)
            posicion.append(1)
        except:
            posicion.append(0)
    return posicion

cadena = 'Esto98es75u,58,5,7,n87string22'
    
isEntero(cadena)
