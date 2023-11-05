from metapopulation_model import EULER_COMPARTIMENTAL_Model
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

'''
sir = SEIR_EULER_Model(3)
sir.set_params([
    np.array([0.2,0.2,0.2]),
    np.array([0.1,0.1,0.1]),
    np.array([0.03,0.03,0.03]),
    np.array([
        [0, 0.1, 0.2],
        [0.2, 0, 0.3],
        [0.13, 0.14, 0]
    ])
])

y0 = np.zeros((5,3))
y0[0] = np.array([999,1499,1999])
y0[1] = np.array([1,1,1])
y0[2] = np.array([0,0,0])
y0[3] = np.array([0,0,0])
y0[4] = np.array([1000,1500,2000])
y0.resize((5*3,))

sir.set_initial_value(0,y0)
sir.solve((0,300))
sir.plot_result('s-1')
'''