import numpy as np

###############################################
##### MODELOS DE SIR, SIS Y SEIR CLÁSICOS #####
###############################################

def fun_sir_model(beta, gamma, N):
    '''
    Genera el sistema de ecuaciones para un modelo SIR.
    
    Parámetros
    ---
    `beta`: Probabilidad de Contagio. Tipo `float`.

    `gamma`: Tasa de Recuperación. Tipo `float`.

    `N`: Cantidad de personas en la población. Tipo `float`.

    Retorno
    ---
    `fun`: Función con el sistema de ecuaciones.
    Tiene por parámetros `t` (variable independiente),
    y `y` (vector de una dimensión, de tamaño 3, para susceptibles, infestados y recuperados).
    '''
    def fun(t,y):
        # Indizado del vector solucion
        # ----------------------------
        # Susceptibles : 0
        # Infestados   : 1
        # Recuperados  : 2
        new_y = np.zeros((3,), dtype=float)
        new_y[0] = - beta * y[0] * y[1] / N
        new_y[1] = beta * y[0] * y[1] / N - gamma * y[1]
        new_y[2] = gamma * y[1]
        return new_y
    return fun

def fun_sis_model(beta, gamma, N):
    '''
    Genera el sistema de ecuaciones para un modelo SIS.
    
    Parámetros
    ---
    `beta`: Probabilidad de Contagio. Tipo `float`.

    `gamma`: Tasa de Recuperación. Tipo `float`.

    `N`: Cantidad de personas en la población. Tipo `float`.

    Retorno
    ---
    `fun`: Función con el sistema de ecuaciones.
    Tiene por parámetros `t` (variable independiente),
    y `y` (vector de una dimensión, de tamaño 2, para susceptibles e infestados).
    '''
    def fun(t,y):
        # Indizado del vector solucion
        # ----------------------------
        # Susceptibles : 0
        # Infestados   : 1
        new_y = np.zeros((2,), dtype=float)
        new_y[0] = - beta * y[0] * y[1] / N + gamma * y[1]
        new_y[1] = beta * y[0] * y[1] / N - gamma * y[1]
        return new_y
    return fun

def fun_seir_model(beta, gamma, sigma, N):
    '''
    Genera el sistema de ecuaciones para un modelo SEIR.
    
    Parámetros
    ---
    `beta`: Probabilidad de Contagio. Tipo `float`

    `gamma`: Tasa de Recuperación. Tipo `float`.

    `sigma`: Tasa de Incubación. Tipo `float`.

    `N`: Cantidad de personas en la población. Tipo `float`.

    Retorno
    ---
    `fun`: Función con el sistema de ecuaciones.
    Tiene por parámetros `t` (variable independiente),
    y `y` (vector de una dimensión, de tamaño 4, para susceptibles, expuestos,
    infestados y recuperados).
    '''
    def fun(t,y):
        # Indizado del vector solucion
        # ----------------------------
        # Susceptibles : 0
        # Expuestos    : 1
        # Infestados   : 2
        # Recuperados  : 3
        new_y = np.zeros((4,), dtype=float)
        new_y[0] = - beta * y[0] * y[2] / N
        new_y[1] = beta * y[0] * y[2] / N - sigma * y[1]
        new_y[2] = sigma * y[1] - gamma * y[2]
        new_y[3] = gamma * y[2]
        return new_y
    return fun

