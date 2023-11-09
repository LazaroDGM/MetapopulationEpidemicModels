import numpy as np
try:
    import numba_models as nm
    from metapopulation_model import EULER_COMPARTIMENTAL_Model, LAGRANGE_COMPARTIMENTAL_Model
except:
    import models.numba_models as nm
    from models.metapopulation_model import EULER_COMPARTIMENTAL_Model, LAGRANGE_COMPARTIMENTAL_Model


class SIR_EULER_Model(EULER_COMPARTIMENTAL_Model):

    def __init__(self, K) -> None:
        super().__init__(['Sus', 'Inf', 'Rec'],
                         ['S', 'I', 'R'],
                         ['Beta', 'Gamma'],
                         K,
                         'SIR EULER')
        
    def fun_SODE(self, t, y):
        return nm.fun_SIR_EULER_SODE(
            t=t,
            y=y,
            F=self._params[0],
            Beta=self._params[1],
            Gamma=self._params[2],
        )
    
    def fun_args_SODE(self, t, y, args):
        return nm.fun_SIR_EULER_SODE(
            t=t,
            y=y,
            F=args[0],
            Beta=args[1],
            Gamma=args[2],
        )

class SIS_EULER_Model(EULER_COMPARTIMENTAL_Model):

    def __init__(self, K) -> None:
        super().__init__(['Susceptibles', 'Infestados'],
                         ['S', 'I'],
                         ['Beta', 'Gamma'],
                         K,
                         'SIS EULER')
        
    def fun_SODE(self, t, y):
        return nm.fun_SIS_EULER_SODE(
            t=t,
            y=y,
            F=self._params[0],
            Beta=self._params[1],
            Gamma=self._params[2],
        )
    
    def fun_args_SODE(self, t, y, args):
        return nm.fun_SIS_EULER_SODE(
            t=t,
            y=y,
            F=args[0],
            Beta=args[1],
            Gamma=args[2],
        )

class SEIR_EULER_Model(EULER_COMPARTIMENTAL_Model):

    def __init__(self, K) -> None:
        super().__init__(['Susceptibles', 'Expuestos', 'Infestados', 'Recuperados'],
                         ['S', 'E', 'I', 'R'],
                         ['Beta', 'Gamma', 'Sigma'],
                         K,
                         'SEIR EULER')
        
    def fun_SODE(self, t, y):
        return nm.fun_SEIR_EULER_SODE(
            t=t,
            y=y,
            F=self._params[0],
            Beta=self._params[1],
            Gamma=self._params[2],
            Sigma=self._params[3],
        )
    
    def fun_args_SODE(self, t, y, args):
        return nm.fun_SEIR_EULER_SODE(
            t=t,
            y=y,
            F=args[0],
            Beta=args[1],
            Gamma=args[2],
            Sigma=args[3],
        )
    
class SIR_LAGRANGE_Model(LAGRANGE_COMPARTIMENTAL_Model):

    def __init__(self, K) -> None:
        super().__init__(['Susceptibles', 'Infestados', 'Recuperados'],
                        ['S', 'I', 'R'],
                        ['Beta', 'Gamma'],
                        K,
                        'SIR LAGRANGE')
        
    def fun_SODE(self, t, y):
        return nm.fun_sir_lagrange(
            t=t,
            y=y,
            Out=self._params[0],
            In=self._params[1],
            Beta=self._params[2],
            Gamma=self._params[3],
        )
    
    def fun_args_SODE(self, t, y, args):
        return nm.fun_sir_lagrange(
            t=t,
            y=y,
            Out=args[0],
            In=args[1],
            Beta=args[2],
            Gamma=args[3],
        )
    
class SIS_LAGRANGE_Model(LAGRANGE_COMPARTIMENTAL_Model):

    def __init__(self, K) -> None:
        super().__init__(['Susceptibles', 'Infestados'],
                         ['S', 'I'],
                         ['Beta', 'Gamma'],
                         K,
                         'SIS LAGRANGE')
        
    def fun_SODE(self, t, y):
        return nm.fun_sis_lagrange(
            t=t,
            y=y,
            Out=self._params[0],
            In=self._params[1],
            Beta=self._params[2],
            Gamma=self._params[3],
        )
    
    def fun_args_SODE(self, t, y, args):
        return nm.fun_sis_lagrange(
            t=t,
            y=y,
            Out=args[0],
            In=args[1],
            Beta=args[2],
            Gamma=args[3],
        )

class SEIR_LAGRANGE_Model(LAGRANGE_COMPARTIMENTAL_Model):

    def __init__(self, K) -> None:
        super().__init__(['Susceptibles', 'Expuestos', 'Infestados', 'Recuperados'],
                         ['S', 'E', 'I', 'R'],
                         ['Beta', 'Gamma', 'Sigma'],
                         K,
                         'SEIR LAGRANGE')
        
    def fun_SODE(self, t, y):
        return nm.fun_seir_lagrange(
            t=t,
            y=y,
            Out=self._params[0],
            In=self._params[1],
            Beta=self._params[2],
            Gamma=self._params[3],
            Sigma=self._params[4],
        )
    
    def fun_args_SODE(self, t, y, args):
        return nm.fun_seir_lagrange(
            t=t,
            y=y,
            Out=args[0],
            In=args[1],
            Beta=args[2],
            Gamma=args[3],
            Sigma=args[4],
        )


sir = SIR_LAGRANGE_Model(3)
sir.set_params([
    np.array([
        [0, 0.1, 0.2],
        [0.2, 0, 0.3],
        [0.13, 0.14, 0]
    ]),
    np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ]),
    30*np.array([0.2,0.2,0.2]),
    30*np.array([0.03,0.03,0.03]),
    30*np.array([0.4,0.4,0.4])
])
'''
y0 = np.zeros((4,3,3))
np.fill_diagonal(y0[0], [999,1499,1999])
np.fill_diagonal(y0[1], [1,1,1])
np.fill_diagonal(y0[2], [0,0,0])
np.fill_diagonal(y0[3], [1000,1500,2000])
y0.resize((4*3*3,))

sir.set_initial_value(0,y0)
sir.solve((0,10))
#sir.plot_result('all')

y = sir._result.y.reshape(4, 3,3, sir._result.y.shape[1])
s = y[0].sum(axis=1)
i = y[1].sum(axis=1)
n = y[3].sum(axis=1)

print((s.flatten() * i.flatten() / n.flatten()).shape)
i_ = np.zeros(i.shape)
i_[:,:i.shape[1]-1] = i[:,1:]
i_ = i_-i
print(i_.flatten())

'''
