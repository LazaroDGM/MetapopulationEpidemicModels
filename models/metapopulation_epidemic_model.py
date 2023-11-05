from metapopulation_model import EULER_COMPARTIMENTAL_Model, LAGRANGE_COMPARTIMENTAL_Model
import numba_models as nm
import numpy as np


class SIR_EULER_Model(EULER_COMPARTIMENTAL_Model):

    def __init__(self, K) -> None:
        super().__init__(['Sus', 'Inf', 'Rec'],
                         ['S', 'I', 'R'],
                         ['Beta', 'Gamma'],
                         K,
                         'SIR EULER')
        
    def fun_SODE(self, t, y, *args):
        return nm.fun_SIR_EULER_SODE(
            t=t,
            y=y,
            Beta=self._params[0],
            Gamma=self._params[1],
            F=self._params[2]
        )

class SIS_EULER_Model(EULER_COMPARTIMENTAL_Model):

    def __init__(self, K) -> None:
        super().__init__(['Susceptibles', 'Infestados'],
                         ['S', 'I'],
                         ['Beta', 'Gamma'],
                         K,
                         'SIS EULER')
        
    def fun_SODE(self, t, y, *args):
        return nm.fun_SIS_EULER_SODE(
            t=t,
            y=y,
            Beta=self._params[0],
            Gamma=self._params[1],
            F=self._params[2]
        )
    
class SEIR_EULER_Model(EULER_COMPARTIMENTAL_Model):

    def __init__(self, K) -> None:
        super().__init__(['Susceptibles', 'Expuestos', 'Infestados', 'Recuperados'],
                         ['S', 'E', 'I', 'R'],
                         ['Beta', 'Gamma', 'Sigma'],
                         K,
                         'SEIR EULER')
        
    def fun_SODE(self, t, y, *args):
        return nm.fun_SEIR_EULER_SODE(
            t=t,
            y=y,
            Beta=self._params[0],
            Gamma=self._params[1],
            Sigma=self._params[2],
            F=self._params[3]
        )
    
class SIR_LAGRANGE_Model(LAGRANGE_COMPARTIMENTAL_Model):

    def __init__(self, K) -> None:
        super().__init__(['Susceptibles', 'Infestados', 'Recuperados'],
                        ['S', 'I', 'R'],
                        ['Beta', 'Gamma'],
                        K,
                        'SIR LAGRANGE')
        
    def fun_SODE(self, t, y, *args):
        return nm.fun_seir_lagrange(
            t=t,
            y=y,
            Beta=self._params[0],
            Gamma=self._params[1],
            Out=self._params[2],
            In=self._params[3],
        )
    
class SIS_LAGRANGE_Model(LAGRANGE_COMPARTIMENTAL_Model):

    def __init__(self, K) -> None:
        super().__init__(['Susceptibles', 'Infestados'],
                         ['S', 'I'],
                         ['Beta', 'Gamma'],
                         K,
                         'SIS LAGRANGE')
        
    def fun_SODE(self, t, y, *args):
        return nm.fun_sis_lagrange(
            t=t,
            y=y,
            Beta=self._params[0],
            Gamma=self._params[1],
            Out=self._params[2],
            In=self._params[3],
        )

class SEIR_LAGRANGE_Model(LAGRANGE_COMPARTIMENTAL_Model):

    def __init__(self, K) -> None:
        super().__init__(['Susceptibles', 'Expuestos', 'Infestados', 'Recuperados'],
                         ['S', 'E', 'I', 'R'],
                         ['Beta', 'Gamma', 'Sigma'],
                         K,
                         'SEIR LAGRANGE')
        
    def fun_SODE(self, t, y, *args):
        return nm.fun_seir_lagrange(
            t=t,
            y=y,
            Beta=self._params[0],
            Gamma=self._params[1],
            Sigma=self._params[2],
            Out=self._params[3],
            In=self._params[4],
        )

'''
sir = SEIR_LAGRANGE_Model(3)
sir.set_params([
    np.array([0.2,0.2,0.2]),
    np.array([0.03,0.03,0.03]),
    np.array([0.4,0.4,0.4]),
    np.array([
        [0, 0.1, 0.2],
        [0.2, 0, 0.3],
        [0.13, 0.14, 0]
    ]),
    np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])
])

y0 = np.zeros((5,3,3))
np.fill_diagonal(y0[0], [999,1499,1999])
np.fill_diagonal(y0[1], [1,1,1])
np.fill_diagonal(y0[2], [0,0,0])
np.fill_diagonal(y0[3], [0,0,0])
np.fill_diagonal(y0[4], [1000,1500,2000])
y0.resize((5*3*3,))

sir.set_initial_value(0,y0)
sir.solve((0,300))
sir.plot_result('i-3', show=False)
sir.plot_result('i-2', show=True)
'''