import numba_models as nm
import numpy as np
from scipy.integrate import solve_ivp

class SODE_Model():

    def __init__(self) -> None:
        self._y0 = None
        self._t0 = None
        self._params = None
        self._params_name = None
        self._result = None

        self._name_model = 'SEDO'

    def fun_SODE(self, t, y, *args) -> np.ndarray:
        '''
        Evaluador del sistema de ecuaciones diferenciales del modelo.
        '''
        raise NotImplementedError('')
    
    def set_params(self, params):
        '''
        Establece los parametros del modelo
        '''
        raise NotImplementedError('')

    def set_initial_value(self, t0, y0):
        '''
        Define los valores iniciales del modelo
        '''
        self._t0 = t0
        self._y0 = y0
    
    def solve(self, t_span, t_eval=None):
        '''
        Resuelve el modelo numericamente
        '''
        try:
            result_sol = solve_ivp(
                fun= self.fun_SODE,
                t_span=t_span,
                y0= self._y0,
                method='RK45',
                t_eval=t_eval
            )
            self._result = result_sol
            return result_sol
        except Exception as e:
            self._result = None
            return e
    
    def plot_result(self):
        raise NotImplementedError()


    def __str__(self) -> str:
        text = f'Modelo: {self._name_model}\nParametros:\n'
        for i in range(len(self._params)):
            text += f'\t- {self._params_name}: '
            text += f'{self._params[i]}\n'
        text += f'Valor Inicial:\n' + \
        f'\t- t0: {self.t0}\n' + \
        f'\t- y0: {self.y0}\n'

