import numba_models as nm
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class SODE_Model():

    def __init__(self, params, params_name, name= 'SEDO', params_correct= False) -> None:

        if len(params) != len(params_name):
            raise Exception('La cantidad de parametros y nombres asociados no coincide')

        self._params = params
        self._params_name = params_name
        self._params_correct = params_correct
        self._name_model = name
        

        self._y0 = None
        self._t0 = None
        self._result = None


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
        if not self._params_correct:
            raise Exception('Los parametros del modelo son Incorrectos')
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
            raise e
    
    def plot_result(self):
        raise NotImplementedError()


    def __repr__(self) -> str:
        text = f'Modelo: {self._name_model}\nParametros:\n'
        for i in range(len(self._params)):
            text += f'\t- {self._params_name[i]}: '
            text += f'{self._params[i]}\n'
        text += f'Valor Inicial:\n' + \
        f'\t- t0: {self._t0}\n' + \
        f'\t- y0: {self._y0}\n'
        return text


class COMPARTMENTAL_Model(SODE_Model):

    def __init__(self, compartmentals_name, compartmentals_short_name, params_name, name) -> None:
        super().__init__([None] * len(params_name), params_name, name, False)

        if len(compartmentals_name) != len(compartmentals_short_name):
            raise Exception('Por cada compartimento debe haber un nombre y un nombre corto')
        
        if len(compartmentals_name) != len(set(compartmentals_name)):
            raise Exception('Nombres de compartimentos repetidos')
        
        if len(compartmentals_short_name)!= len(set(compartmentals_short_name)):
            raise Exception('Nombres Cortos de compartimentos repetidos')
        
        
        self.__compartmentals_name = compartmentals_name
        self.__compartmentals_short_name = [str.lower(i) for i in compartmentals_short_name]
        self.__total_compartmentals = len(compartmentals_name)

    def set_params(self, params: dict|list):
        try:
            if params is dict:
                for i, name in enumerate(self._params_name):
                    self._params[i] = params[name]
                self._params_correct = True
            else:
                for i in range(len(self._params_name)):
                    self._params[i] = params[i]
                self._params_correct = True
        except Exception as e:
            self._params_correct = False
            raise(e)

    def plot_result(self,mode='total',
                    xlabel='Tiempo (dias)',
                    ylabel= 'Cantidad de personas (unidades)',
                    show= True,
                    grid= True):

        if self._result is not None:
            mode = mode.lower()
            t = self._result.t
            ys = self._result.y
            if mode == 'total':
                for i, y in enumerate(ys):
                    plt.plot(t, y, label= self.__compartmentals_name[i])
            else:
                try:
                    index = self.__compartmentals_short_name.index(mode)
                    plt.plot(t, ys[index], label= self.__compartmentals_name[index])
                except:
                    raise Exception('Modo de graficado incorrecto')
            if show:
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(label=self._name_model)
                if grid:
                    plt.grid()
                plt.legend()
                plt.show()
        else:
            raise Exception('El modelo no esta resuelto.')
