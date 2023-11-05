import numpy as np
from numba import jit, njit

###############################################
##### MODELOS DE SIR, SIS Y SEIR CLÁSICOS #####
###############################################

@njit
def fun_SIR_SODE(t, y, beta, gamma, N):
    
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

@njit
def fun_SIS_SODE(t, y, beta, gamma, N):

    # Indizado del vector solucion
    # ----------------------------
    # Susceptibles : 0
    # Infestados   : 1
    new_y = np.zeros((2,), dtype=float)
    new_y[0] = - beta * y[0] * y[1] / N + gamma * y[1]
    new_y[1] = beta * y[0] * y[1] / N - gamma * y[1]
    return new_y

@njit
def fun_SEIR_SODE(t, y, beta, gamma, sigma, N):

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

#######################################################
##### MOVIMIENTO EULERIANO Y LAGRANGIANO (PUROS) ######
#######################################################

@njit
def fun_EULER_SODE(t,y, F):
    
    K = F.shape[0]
    new_y = np.zeros((K,), dtype=float)
    for i in range(K):
        for j in range(K):
            new_y[i] += F[j,i] * y[j] - F[i,j] * y[i]
    return new_y

@njit
def fun_LAGRANGE_SODE(t,y, O, I):

    K = I.shape[0]
    y = np.resize(y, (K,K))
    new_y = np.zeros((K,K), dtype=float)
    for i in range(K):
        for j in range(K):
            new_y[i,i] += I[i,j] * y[i,j] - O[i,j] * y[i,i]
            if i != j:
                new_y[i,j] = O[i,j] * y[i,i] - I[i,j] * y[i,j]
    return new_y.flatten()

#################################################
###### MODELO SIR CON MOVIMIENTO EULERIANO ######
#################################################

@njit
def fun_SIR_EULER_SODE(t, y, F, Beta, Gamma):

    K = F.shape[0]

    y = y.reshape((4,K))
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
    new_y = new_y.reshape((4*K,))
    return new_y

###################################################
###### MODELO SIR CON MOVIMIENTO LAGRANGIANO ######
###################################################

@njit
def fun_sir_lagrange(t,y, Out, In, Beta, Gamma):

    K = Out.shape[0]
    Out_i_k = Out.sum(axis=1)

    y = y.reshape((4,K,K))
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
    new_y = new_y.reshape((4*K*K,))
    return new_y

#################################################
###### MODELO SIS CON MOVIMIENTO EULERIANO ######
#################################################

@njit
def fun_SIS_EULER_SODE(t,y, F, Beta, Gamma):

    K = F.shape[0]
    y = y.reshape((3,K))
    new_y = np.zeros((3,K))
    for i in range(K):
        new_y[0,i] = - Beta[i] * y[0,i] * y[1,i] / y[2,i] + Gamma[i] * y[1,i]
        new_y[1,i] = Beta[i] * y[0,i] * y[1,i] / y[2,i] - Gamma[i] * y[1,i]
        for j in range(K):
            new_y[0,i] += F[j,i] * y[0,j] - F[i,j] * y[0,i]
            new_y[1,i] += F[j,i] * y[1,j] - F[i,j] * y[1,i]
            new_y[2,i] += F[j,i] * y[2,j] - F[i,j] * y[2,i]
    new_y = new_y.reshape((3*K,))
    return new_y

###################################################
###### MODELO SIS CON MOVIMIENTO LAGRANGIANO ######
###################################################

@njit
def fun_sis_lagrange(t,y, Out, In, Beta, Gamma):

    K = Out.shape[0]
    Out_i_k = Out.sum(axis=1)

    y = y.reshape((3,K,K))
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
    new_y = new_y.reshape((3*K*K,))
    return new_y

##################################################
###### MODELO SEIR CON MOVIMIENTO EULERIANO ######
##################################################

@njit
def fun_SEIR_EULER_SODE(t,y, F, Beta, Gamma, Sigma):

    K = F.shape[0]
    y = y.reshape((5,K))
    new_y = np.zeros((5,K))
    for i in range(K):
        new_y[0,i] = - Beta[i] * y[0,i] * y[1,i] / y[4,i]
        new_y[1,i] = Beta[i] * y[0,i] * y[1,i] / y[4,i] - Sigma[i] * y[1,i]
        new_y[2,i] = Sigma[i] * y[1,i] - Gamma[i] * y[2,i]
        new_y[3,i] = Gamma[i] * y[2,i]
        for j in range(K):
            new_y[0,i] += F[j,i] * y[0,j] - F[i,j] * y[0,i]
            new_y[1,i] += F[j,i] * y[1,j] - F[i,j] * y[1,i]
            new_y[2,i] += F[j,i] * y[2,j] - F[i,j] * y[2,i]
            new_y[3,i] += F[j,i] * y[3,j] - F[i,j] * y[3,i]
            new_y[4,i] += F[j,i] * y[4,j] - F[i,j] * y[4,i]
    new_y = new_y.reshape((5*K,))
    return new_y

####################################################
###### MODELO SEIR CON MOVIMIENTO LAGRANGIANO ######
####################################################

@njit
def fun_seir_lagrange(t,y, Out, In, Beta, Gamma, Sigma):
    
    K = Out.shape[0]
    Out_i_k = Out.sum(axis=1)

    y = y.reshape((5,K,K))
    I_k_i = y[2].sum(axis=0)
    N_k_i = y[4].sum(axis=0)
    new_y = np.zeros((5,K,K))
    for i in range(K):
        for j in range(K):
            if i == j:
                new_y[0,i,i] = - Beta[i] * y[0,i,i] * I_k_i[i] / N_k_i[i] - \
                                y[0,i,i] * Out_i_k[i] + (In[i,] * y[0,i]).sum()
                new_y[1,i,i] =   Beta[i] * y[0,i,i] * I_k_i[i] / N_k_i[i] - Sigma[i] * y[1,i,i] - \
                                y[1,i,i] * Out_i_k[i] + (In[i,] * y[1,i]).sum()
                new_y[2,i,i] =   Sigma[i] * y[1,i,i] - Gamma[i] * y[2,i,i] - \
                                y[2,i,i] * Out_i_k[i] + (In[i,] * y[2,i]).sum()
                new_y[3,i,i] =   Gamma[i] * y[2,i,i] - \
                                y[3,i,i] * Out_i_k[i] + (In[i,] * y[3,i]).sum()
                new_y[4,i,i] = - y[4,i,i] * Out_i_k[i] + (In[i,] * y[4,i]).sum()
            else:
                new_y[0,i,j] = - Beta[j] * y[0,i,j] * I_k_i[j] / N_k_i[j] - \
                                In[i,j] * y[0,i,j] + Out[i,j] * y[0,i,i]
                new_y[1,i,j] =   Beta[j] * y[0,i,j] * I_k_i[j] / N_k_i[j] - Sigma[j] * y[1,i,j] - \
                                In[i,j] * y[1,i,j] + Out[i,j] * y[1,i,i]
                new_y[2,i,j] =   Sigma[j] * y[1,i,j] - Gamma[j] * y[2,i,j] - \
                                In[i,j] * y[2,i,j] + Out[i,j] * y[2,i,i]
                new_y[3,i,j] =   Gamma[j] * y[2,i,j] - \
                                In[i,j] * y[3,i,j] + Out[i,j] * y[3,i,i]
                new_y[4,i,j] = - In[i,j] * y[4,i,j] + Out[i,j] * y[4,i,i]
                                
    new_y = new_y.reshape((5*K*K,))
    return new_y