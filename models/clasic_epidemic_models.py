from SODE_model import COMPARTMENTAL_Model
import numpy as np
import numba_models as nm
import matplotlib.pyplot as plt

class SIR_Model(COMPARTMENTAL_Model):

    def __init__(self) -> None:
        super().__init__(['Susceptibles', 'Infestados', 'Recuperados'],
                        ['S', 'I', 'R'], ['Beta', 'Gamma', 'N'],'SIR')
        
    def fun_SODE(self, t, y) -> np.ndarray:
        return nm.fun_SIR_SODE(t,y,
            beta= self._params[0],
            gamma= self._params[1],
            N= self._params[2],
        )


class SIS_Model(COMPARTMENTAL_Model):

    def __init__(self) -> None:
        super().__init__(['Susceptibles', 'Infestados'],
                        ['S', 'I'], ['Beta', 'Gamma', 'N'],'SIS')
        
    def fun_SODE(self, t, y) -> np.ndarray:
        return nm.fun_SIS_SODE(t,y,
            beta= self._params[0],
            gamma= self._params[1],
            N= self._params[2],
        )

class SEIR_Model(COMPARTMENTAL_Model):

    def __init__(self) -> None:
        super().__init__(['Susceptibles', 'Expuestos', 'Infestados', 'Recuperados'],
                        ['S', 'E', 'I', 'R'], ['Beta', 'Gamma', 'Sigma', 'N'],'SEIR')
        
    def fun_SODE(self, t, y) -> np.ndarray:
        return nm.fun_SEIR_SODE(t,y,
            beta= self._params[0],
            gamma= self._params[1],
            sigma= self._params[2],
            N= self._params[3],
        )

