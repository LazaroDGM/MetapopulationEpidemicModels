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

#######################################################
##### MOVIMIENTO EULERIANO Y LAGRANGIANO (PUROS) ######
#######################################################

def fun_euler_mov(F):
    '''
    Genera el sistema de ecuaciones para un modelo movimiento euleriano.
    
    Parámetros
    ---
    `F`: Matriz de movimiento de dimensión `K x K` (matriz cuadrada con diagonal nula).
    Indica la tasa de traslado de un nodo `i` a un nodo `j`.

    Retorno
    ---
    `fun`: Función con el sistema de ecuaciones.
    Tiene por parámetros `t` (variable independiente),
    y `y` (vector de una dimensión, de tamaño `K`, con la población inicial de cada nodo).
    '''
    K = F.shape[0]
    def fun(t,y):
        new_y = np.zeros((K,), dtype=float)
        for i in range(K):
            for j in range(K):
                new_y[i] += F[j,i] * y[j] - F[i,j] * y[i]
        return new_y
    return fun

def fun_lagrange_mov(O, I):
    '''
    Genera el sistema de ecuaciones para un modelo movimiento lagrangiano.
    
    Parámetros
    ---
    `O`: Matriz de movimiento de emigración de dimensión `K x K` (matriz cuadrada con diagonal nula).
    Indica la tasa de traslado de los agentes de `i` en `i` que se mueven al nodo `j`.

    `I`: Matriz de movimiento de inmigración de dimensión `K x K` (matriz cuadrada con diagonal nula).
    Indica la tasa de traslado de los agentes de `i` en `j` que regresan al nodo `i`.

    Retorno
    ---
    `fun`: Función con el sistema de ecuaciones.
    Tiene por parámetros `t` (variable independiente),
    y `y` (vector de dimensión `K x K`, con la población inicial en cada nodo `i` que se encuentran en el nodo `j`).
    '''
    K = I.shape[0]
    def fun(t, y):
        y = np.resize(y, (K,K))
        new_y = np.zeros((K,K), dtype=float)
        for i in range(K):
            for j in range(K):
                new_y[i,i] += I[i,j] * y[i,j] - O[i,j] * y[i,i]
                if i != j:
                    new_y[i,j] = O[i,j] * y[i,i] - I[i,j] * y[i,j]
        return new_y.flatten()
    return fun

#################################################
###### MODELO SIR CON MOVIMIENTO EULERIANO ######
#################################################

def fun_sir_eulerian(F, Beta, Gamma):
    '''
    Genera el sistema de ecuaciones para un modelo SIR movimiento euleriano.
    
    Parámetros
    ---
    `F`: Matriz de movimiento de dimensión `K x K` (matriz cuadrada con diagonal nula).
    Indica la tasa de traslado de un nodo `i` a un nodo `j`.

    `Beta`: Probabilidad de Contagio por nodo. Vector de tipo `float` y tamaño `K`.

    `Gamma`: Tasa de Recuperación por nodo. Vector de tipo `float` y tamaño `K`.

    Retorno
    ---
    `fun`: Función con el sistema de ecuaciones.
    Tiene por parámetros `t` (variable independiente),
    y `y` con la siguiente estructura: Vector de tamaño `4*K` (aplanado),
    con los susceptibles iniciales por cada nodo, más los infestados, los recuperados
    y la población total, cada uno por cada nodo.
    '''
    K = F.shape[0]
    def fun(t,y):
        y.resize((4,K))
        new_y = np.zeros((4,K))
        for i in range(K):
            new_y[0,i] = - Beta[i] * y[0,i] * y[1,i] / y[3,i]
            new_y[1,i] = Beta[i] * y[0,i] * y[1,i] / y[3,i] - Gamma[i] * y[1,i]
            new_y[2,i] = Gamma[i] * y[1,i]
            for j in range(K):
                new_y[0,i] += F[j,i] * y[0,j] - F[i,j] * y[0,i]
                new_y[1,i] += F[j,i] * y[1,j] - F[i,j] * y[1,i]
                new_y[2,i] += F[j,i] * y[2,j] - F[i,j] * y[2,i]
                new_y[3,i] += F[j,i] * y[3,j] - F[i,j] * y[3,i]
        y.resize((4*K,))
        new_y.resize((4*K,))
        return new_y
    return fun

def fun_sir_eulerian_lite(F, Beta, Gamma):
    '''
    Genera el sistema de ecuaciones para un modelo SIR movimiento euleriano.
    No contempla las ecuaciones diferenciales destinadas para la variación de
    la población total por nodo. Se asume que `N_i_0 = S_i_0 + I_i_0 + R_i_0`.
    
    Parámetros
    ---
    `F`: Matriz de movimiento de dimensión `K x K` (matriz cuadrada con diagonal nula).
    Indica la tasa de traslado de un nodo `i` a un nodo `j`.

    `Beta`: Probabilidad de Contagio por nodo. Vector de tipo `float` y tamaño `K`.

    `Gamma`: Tasa de Recuperación por nodo. Vector de tipo `float` y tamaño `K`.

    Retorno
    ---
    `fun`: Función con el sistema de ecuaciones.
    Tiene por parámetros `t` (variable independiente),
    y `y` con la siguiente estructura: Vector de tamaño `3*K` (aplanado),
    con los susceptibles iniciales por cada nodo, más los infestados, y los recuperados,
    cada uno por cada nodo.
    '''
    K = F.shape[0]
    def fun(t,y):
        y.resize((3,K))
        new_y = np.zeros((3,K))
        for i in range(K):
            N_i = float(y[0,i] + y[1,i] + y[2,i])
            new_y[0,i] = - Beta[i] * y[0,i] * y[1,i] / N_i
            new_y[1,i] = Beta[i] * y[0,i] * y[1,i] / N_i - Gamma[i] * y[1,i]
            new_y[2,i] = Gamma[i] * y[1,i]
            for j in range(K):
                new_y[0,i] += F[j,i] * y[0,j] - F[i,j] * y[0,i]
                new_y[1,i] += F[j,i] * y[1,j] - F[i,j] * y[1,i]
                new_y[2,i] += F[j,i] * y[2,j] - F[i,j] * y[2,i]
        y.resize((3*K,))
        new_y.resize((3*K,))
        return new_y
    return fun

###################################################
###### MODELO SIR CON MOVIMIENTO LAGRANGIANO ######
###################################################

def fun_sir_lagrange(Out, In, Beta, Gamma):
    '''
    Genera el sistema de ecuaciones para un modelo SIR movimiento lagrangiano.
    
    Parámetros
    ---
    `Out`: Matriz de movimiento de emigración de dimensión `K x K` (matriz cuadrada con diagonal nula).
    Indica la tasa de traslado de los agentes de `i` en `i` que se mueven al nodo `j`.

    `In`: Matriz de movimiento de inmigración de dimensión `K x K` (matriz cuadrada con diagonal nula).
    Indica la tasa de traslado de los agentes de `i` en `j` que regresan al nodo `i`.

    `Beta`: Probabilidad de Contagio por nodo. Vector de tipo `float` y tamaño `K`.

    `Gamma`: Tasa de Recuperación por nodo. Vector de tipo `float` y tamaño `K`.

    Retorno
    ---
    `fun`: Función con el sistema de ecuaciones.
    Tiene por parámetros `t` (variable independiente),
    y `y` con la siguiente estructura: Vector de tamaño `4*K*K` (aplanado),
    con los susceptibles iniciales, los infestados, los recuperados
    y la población total, de un nodo en otro.
    '''
    K = Out.shape[0]
    Out_i_k = Out.sum(axis=1)
    def fun(t,y):
        y.resize((4,K,K))
        I_k_i = y[1].sum(axis=0)
        N_k_i = y[3].sum(axis=0)
        new_y = np.zeros((4,K,K))
        for i in range(K):
            for j in range(K):
                if i == j:
                    new_y[0,i,i] = - Beta[i] * y[0,i,i] * I_k_i[i] / N_k_i[i] - \
                                    y[0,i,i] * Out_i_k[i] + (In[i,] * y[0,i]).sum()
                    new_y[1,i,i] =   Beta[i] * y[0,i,i] * I_k_i[i] / N_k_i[i] - Gamma[i] * y[1,i,i] - \
                                    y[1,i,i] * Out_i_k[i] + (In[i,] * y[1,i]).sum()
                    new_y[2,i,i] =   Gamma[i] * y[1,i,i] - \
                                    y[2,i,i] * Out_i_k[i] + (In[i,] * y[2,i]).sum()
                    new_y[3,i,i] = - y[3,i,i] * Out_i_k[i] + (In[i,] * y[3,i]).sum()
                else:
                    new_y[0,i,j] = - Beta[j] * y[0,i,j] * I_k_i[j] / N_k_i[j] - \
                                    In[i,j] * y[0,i,j] + Out[i,j] * y[0,i,i]
                    new_y[1,i,j] =   Beta[j] * y[0,i,j] * I_k_i[j] / N_k_i[j] - Gamma[j] * y[1,i,j] - \
                                    In[i,j] * y[1,i,j] + Out[i,j] * y[1,i,i]
                    new_y[2,i,j] =   Gamma[j] * y[1,i,j] - \
                                    In[i,j] * y[2,i,j] + Out[i,j] * y[2,i,i]
                    new_y[3,i,j] = - In[i,j] * y[3,i,j] + Out[i,j] * y[3,i,i]
                                   
        y.resize((4*K*K,))
        new_y.resize((4*K*K,))
        return new_y
    return fun

def fun_sir_lagrange_lite(Out, In, Beta, Gamma):
    '''
    Genera el sistema de ecuaciones para un modelo SIR movimiento lagrangiano.
    No contempla las ecuaciones diferenciales destinadas para la variación de
    la población total por nodo.

    Parámetros
    ---
    `Out`: Matriz de movimiento de emigración de dimensión `K x K` (matriz cuadrada con diagonal nula).
    Indica la tasa de traslado de los agentes de `i` en `i` que se mueven al nodo `j`.

    `In`: Matriz de movimiento de inmigración de dimensión `K x K` (matriz cuadrada con diagonal nula).
    Indica la tasa de traslado de los agentes de `i` en `j` que regresan al nodo `i`.

    `Beta`: Probabilidad de Contagio por nodo. Vector de tipo `float` y tamaño `K`.

    `Gamma`: Tasa de Recuperación por nodo. Vector de tipo `float` y tamaño `K`.

    Retorno
    ---
    `fun`: Función con el sistema de ecuaciones.
    Tiene por parámetros `t` (variable independiente),
    y `y` con la siguiente estructura: Vector de tamaño `3*K*K` (aplanado),
    con los susceptibles iniciales, los infestados, y los recuperados
    , de un nodo en otro.
    '''
    K = Out.shape[0]
    Out_i_k = Out.sum(axis=1)
    def fun(t,y):
        y.resize((3,K,K))
        I_k_i = y[1].sum(axis=0)
        N_k_i = (y[0] + y[1] + y[2]).sum(axis=0)
        new_y = np.zeros((3,K,K))
        for i in range(K):
            for j in range(K):
                if i == j:
                    new_y[0,i,i] = - Beta[i] * y[0,i,i] * I_k_i[i] / N_k_i[i] - \
                                    y[0,i,i] * Out_i_k[i] + (In[i,] * y[0,i]).sum()
                    new_y[1,i,i] =   Beta[i] * y[0,i,i] * I_k_i[i] / N_k_i[i] - Gamma[i] * y[1,i,i] - \
                                    y[1,i,i] * Out_i_k[i] + (In[i,] * y[1,i]).sum()
                    new_y[2,i,i] =   Gamma[i] * y[1,i,i] - \
                                    y[2,i,i] * Out_i_k[i] + (In[i,] * y[2,i]).sum()
                else:
                    new_y[0,i,j] = - Beta[j] * y[0,i,j] * I_k_i[j] / N_k_i[j] - \
                                    In[i,j] * y[0,i,j] + Out[i,j] * y[0,i,i]
                    new_y[1,i,j] =   Beta[j] * y[0,i,j] * I_k_i[j] / N_k_i[j] - Gamma[j] * y[1,i,j] - \
                                    In[i,j] * y[1,i,j] + Out[i,j] * y[1,i,i]
                    new_y[2,i,j] =   Gamma[j] * y[1,i,j] - \
                                    In[i,j] * y[2,i,j] + Out[i,j] * y[2,i,i]            
        y.resize((3*K*K,))
        new_y.resize((3*K*K,))
        return new_y
    return fun

#################################################
###### MODELO SIS CON MOVIMIENTO EULERIANO ######
#################################################

def fun_sis_eulerian(F, Beta, Gamma):
    '''
    Genera el sistema de ecuaciones para un modelo SIS movimiento euleriano.
    
    Parámetros
    ---
    `F`: Matriz de movimiento de dimensión `K x K` (matriz cuadrada con diagonal nula).
    Indica la tasa de traslado de un nodo `i` a un nodo `j`.

    `Beta`: Probabilidad de Contagio por nodo. Vector de tipo `float` y tamaño `K`.

    `Gamma`: Tasa de Recuperación por nodo. Vector de tipo `float` y tamaño `K`.

    Retorno
    ---
    `fun`: Función con el sistema de ecuaciones.
    Tiene por parámetros `t` (variable independiente),
    y `y` con la siguiente estructura: Vector de tamaño `3*K` (aplanado),
    con los susceptibles iniciales por cada nodo, más los infestados
    y la población total, cada uno por cada nodo.
    '''
    K = F.shape[0]
    def fun(t,y):
        y.resize((3,K))
        new_y = np.zeros((3,K))
        for i in range(K):
            new_y[0,i] = - Beta[i] * y[0,i] * y[1,i] / y[2,i] + Gamma[i] * y[1,i]
            new_y[1,i] = Beta[i] * y[0,i] * y[1,i] / y[2,i] - Gamma[i] * y[1,i]
            for j in range(K):
                new_y[0,i] += F[j,i] * y[0,j] - F[i,j] * y[0,i]
                new_y[1,i] += F[j,i] * y[1,j] - F[i,j] * y[1,i]
                new_y[2,i] += F[j,i] * y[2,j] - F[i,j] * y[2,i]
        y.resize((3*K,))
        new_y.resize((3*K,))
        return new_y
    return fun

def fun_sis_eulerian_lite(F, Beta, Gamma):
    '''
    Genera el sistema de ecuaciones para un modelo SIS movimiento euleriano.
    No contempla las ecuaciones diferenciales destinadas para la variación de
    la población total por nodo. Se asume que `N_i_0 = S_i_0 + I_i_0`.
    
    Parámetros
    ---
    `F`: Matriz de movimiento de dimensión `K x K` (matriz cuadrada con diagonal nula).
    Indica la tasa de traslado de un nodo `i` a un nodo `j`.

    `Beta`: Probabilidad de Contagio por nodo. Vector de tipo `float` y tamaño `K`.

    `Gamma`: Tasa de Recuperación por nodo. Vector de tipo `float` y tamaño `K`.

    Retorno
    ---
    `fun`: Función con el sistema de ecuaciones.
    Tiene por parámetros `t` (variable independiente),
    y `y` con la siguiente estructura: Vector de tamaño `2*K` (aplanado),
    con los susceptibles iniciales por cada nodo, más los infestados,
    cada uno por cada nodo.
    '''
    K = F.shape[0]
    def fun(t,y):
        y.resize((2,K))
        new_y = np.zeros((2,K))
        for i in range(K):
            N_i = y[0,i] + y[1,i]
            new_y[0,i] = - Beta[i] * y[0,i] * y[1,i] / N_i + Gamma[i] * y[1,i]
            new_y[1,i] = Beta[i] * y[0,i] * y[1,i] / N_i - Gamma[i] * y[1,i]
            for j in range(K):
                new_y[0,i] += F[j,i] * y[0,j] - F[i,j] * y[0,i]
                new_y[1,i] += F[j,i] * y[1,j] - F[i,j] * y[1,i]
        y.resize((2*K,))
        new_y.resize((2*K,))
        return new_y
    return fun

###################################################
###### MODELO SIS CON MOVIMIENTO LAGRANGIANO ######
###################################################

def fun_sis_lagrange(Out, In, Beta, Gamma):
    '''
    Genera el sistema de ecuaciones para un modelo SIS movimiento lagrangiano.
    
    Parámetros
    ---
    `Out`: Matriz de movimiento de emigración de dimensión `K x K` (matriz cuadrada con diagonal nula).
    Indica la tasa de traslado de los agentes de `i` en `i` que se mueven al nodo `j`.

    `In`: Matriz de movimiento de inmigración de dimensión `K x K` (matriz cuadrada con diagonal nula).
    Indica la tasa de traslado de los agentes de `i` en `j` que regresan al nodo `i`.

    `Beta`: Probabilidad de Contagio por nodo. Vector de tipo `float` y tamaño `K`.

    `Gamma`: Tasa de Recuperación por nodo. Vector de tipo `float` y tamaño `K`.

    Retorno
    ---
    `fun`: Función con el sistema de ecuaciones.
    Tiene por parámetros `t` (variable independiente),
    y `y` con la siguiente estructura: Vector de tamaño `3*K*K` (aplanado),
    con los susceptibles iniciales, los infestados
    y la población total, de un nodo en otro.
    '''
    K = Out.shape[0]
    Out_i_k = Out.sum(axis=1)
    def fun(t,y):
        y.resize((3,K,K))
        I_k_i = y[1].sum(axis=0)
        N_k_i = y[2].sum(axis=0)
        new_y = np.zeros((3,K,K))
        for i in range(K):
            for j in range(K):
                if i == j:
                    new_y[0,i,i] = - Beta[i] * y[0,i,i] * I_k_i[i] / N_k_i[i] + Gamma[i] * y[1,i,i] - \
                                    y[0,i,i] * Out_i_k[i] + (In[i,] * y[0,i]).sum()
                    new_y[1,i,i] =   Beta[i] * y[0,i,i] * I_k_i[i] / N_k_i[i] - Gamma[i] * y[1,i,i] - \
                                    y[1,i,i] * Out_i_k[i] + (In[i,] * y[1,i]).sum()
                    new_y[2,i,i] = - y[2,i,i] * Out_i_k[i] + (In[i,] * y[2,i]).sum()
                else:
                    new_y[0,i,j] = - Beta[j] * y[0,i,j] * I_k_i[j] / N_k_i[j] + Gamma[j] * y[1,i,j] - \
                                    In[i,j] * y[0,i,j] + Out[i,j] * y[0,i,i]
                    new_y[1,i,j] =   Beta[j] * y[0,i,j] * I_k_i[j] / N_k_i[j] - Gamma[j] * y[1,i,j] - \
                                    In[i,j] * y[1,i,j] + Out[i,j] * y[1,i,i]
                    new_y[2,i,j] = - In[i,j] * y[2,i,j] + Out[i,j] * y[2,i,i]
                                   
        y.resize((3*K*K,))
        new_y.resize((3*K*K,))
        return new_y
    return fun

def fun_sis_lagrange_lite(Out, In, Beta, Gamma):
    '''
    Genera el sistema de ecuaciones para un modelo SIS movimiento lagrangiano.
    No contempla las ecuaciones diferenciales destinadas para la variación de
    la población total por nodo.

    Parámetros
    ---
    `Out`: Matriz de movimiento de emigración de dimensión `K x K` (matriz cuadrada con diagonal nula).
    Indica la tasa de traslado de los agentes de `i` en `i` que se mueven al nodo `j`.

    `In`: Matriz de movimiento de inmigración de dimensión `K x K` (matriz cuadrada con diagonal nula).
    Indica la tasa de traslado de los agentes de `i` en `j` que regresan al nodo `i`.

    `Beta`: Probabilidad de Contagio por nodo. Vector de tipo `float` y tamaño `K`.

    `Gamma`: Tasa de Recuperación por nodo. Vector de tipo `float` y tamaño `K`.

    Retorno
    ---
    `fun`: Función con el sistema de ecuaciones.
    Tiene por parámetros `t` (variable independiente),
    y `y` con la siguiente estructura: Vector de tamaño `2*K*K` (aplanado),
    con los susceptibles iniciales,y los infestados
    , de un nodo en otro.
    '''
    K = Out.shape[0]
    Out_i_k = Out.sum(axis=1)
    def fun(t,y):
        y.resize((2,K,K))
        I_k_i = y[1].sum(axis=0)
        N_k_i = (y[0] + y[1]).sum(axis=0)
        new_y = np.zeros((2,K,K))
        for i in range(K):
            for j in range(K):
                if i == j:
                    new_y[0,i,i] = - Beta[i] * y[0,i,i] * I_k_i[i] / N_k_i[i] + Gamma[i] * y[1,i,i] - \
                                    y[0,i,i] * Out_i_k[i] + (In[i,] * y[0,i]).sum()
                    new_y[1,i,i] =   Beta[i] * y[0,i,i] * I_k_i[i] / N_k_i[i] - Gamma[i] * y[1,i,i] - \
                                    y[1,i,i] * Out_i_k[i] + (In[i,] * y[1,i]).sum()
                else:
                    new_y[0,i,j] = - Beta[j] * y[0,i,j] * I_k_i[j] / N_k_i[j] + Gamma[j] * y[1,i,j] - \
                                    In[i,j] * y[0,i,j] + Out[i,j] * y[0,i,i]
                    new_y[1,i,j] =   Beta[j] * y[0,i,j] * I_k_i[j] / N_k_i[j] - Gamma[j] * y[1,i,j] - \
                                    In[i,j] * y[1,i,j] + Out[i,j] * y[1,i,i]        
        y.resize((2*K*K,))
        new_y.resize((2*K*K,))
        return new_y
    return fun

