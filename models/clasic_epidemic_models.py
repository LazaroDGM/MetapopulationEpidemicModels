from SODE_model import SODE_Model
import numpy as np
import numba_models as nm
import matplotlib.pyplot as plt

class SIR_Model(SODE_Model):

    def __init__(self) -> None:
        super().__init__()
        self._params = [None]*3
        self._params_name = ['Beta', 'Gamma', 'N']
        self._params_correct = False

        self._name_model = 'SIR'

    def set_params(self, params: dict):
        try:
            if params is dict:
                self._params[0] = params['Beta']
                self._params[1] = params['Gamma']
                self._params[2] = params['N']
                self._params_correct = True
            else:
                self._params[0] = params[0]
                self._params[1] = params[1]
                self._params[2] = params[2]
                self._params_correct = True
        except Exception as e:
            self._params_correct = False
            raise(e)
        
    def fun_SODE(self, t, y) -> np.ndarray:
        if self._params_correct:
            return nm.fun_SIR_SODE(t,y,
                beta= self._params[0],
                gamma= self._params[1],
                N= self._params[2],
            )
        else:
            raise Exception('Los parametros del modelo son Incorrectos')
        
    def plot_result(self,mode='t', show= True):
        if self._result is not None:
            mode = mode.lower()
            t = self._result.t
            ys = self._result.y
            if mode == 't':
                for y in self._result.y:
                    plt.plot(t, y)
            elif mode == 's':
                plt.plot(t, ys[0])
            elif mode == 'i':
                plt.plot(t, ys[1])
            elif mode == 'r':
                plt.plot(t, ys[2])
            else:
                raise Exception('Modo de graficado incorrecto')
            if show:
                plt.show()
        else:
            raise Exception('El modelo no esta resuelto.')
    
    def set_initial_value(self, t0, y0):
        super().set_initial_value(t0, y0)


class SIS_Model(SODE_Model):

    def __init__(self) -> None:
        super().__init__()
        self._params = [None]*3
        self._params_name = ['Beta', 'Gamma', 'N']
        self._params_correct = False

        self._name_model = 'SIS'

    def set_params(self, params: dict):
        try:
            if params is dict:
                self._params[0] = params['Beta']
                self._params[1] = params['Gamma']
                self._params[2] = params['N']
                self._params_correct = True
            else:
                self._params[0] = params[0]
                self._params[1] = params[1]
                self._params[2] = params[2]
                self._params_correct = True
        except Exception as e:
            self._params_correct = False
            raise(e)
        
    def fun_SODE(self, t, y) -> np.ndarray:
        if self._params_correct:
            return nm.fun_SIS_SODE(t,y,
                beta= self._params[0],
                gamma= self._params[1],
                N= self._params[2],
            )
        else:
            raise Exception('Los parametros del modelo son Incorrectos')
        
    def plot_result(self,mode='t', show= True):
        if self._result is not None:
            mode = mode.lower()
            t = self._result.t
            ys = self._result.y
            if mode == 't':
                for y in self._result.y:
                    plt.plot(t, y)
            elif mode == 's':
                plt.plot(t, ys[0])
            elif mode == 'i':
                plt.plot(t, ys[1])
            else:
                raise Exception('Modo de graficado incorrecto')
            if show:
                plt.show()
        else:
            raise Exception('El modelo no esta resuelto.')
    
    def set_initial_value(self, t0, y0):
        super().set_initial_value(t0, y0)

class SEIR_Model(SODE_Model):

    def __init__(self) -> None:
        super().__init__()
        self._params = [None]*4
        self._params_name = ['Beta', 'Gamma', 'N']
        self._params_correct = False

        self._name_model = 'SIS'

    def set_params(self, params: dict):
        try:
            if params is dict:
                self._params[0] = params['Beta']
                self._params[1] = params['Gamma']
                self._params[2] = params['Sigma']
                self._params[3] = params['N']
                self._params_correct = True
            else:
                self._params[0] = params[0]
                self._params[1] = params[1]
                self._params[2] = params[2]
                self._params[3] = params[3]
                self._params_correct = True
        except Exception as e:
            self._params_correct = False
            raise(e)
        
    def fun_SODE(self, t, y) -> np.ndarray:
        if self._params_correct:
            return nm.fun_SEIR_SODE(t,y,
                beta= self._params[0],
                gamma= self._params[1],
                sigma= self._params[2],
                N= self._params[3],
            )
        else:
            raise Exception('Los parametros del modelo son Incorrectos')
        
    def plot_result(self,mode='t', show= True):
        if self._result is not None:
            mode = mode.lower()
            t = self._result.t
            ys = self._result.y
            if mode == 't':
                for y in self._result.y:
                    plt.plot(t, y)
            elif mode == 's':
                plt.plot(t, ys[0])
            elif mode == 'e':
                plt.plot(t, ys[1])
            elif mode == 'i':
                plt.plot(t, ys[2])
            elif mode == 'r':
                plt.plot(t, ys[3])
            else:
                raise Exception('Modo de graficado incorrecto')
            if show:
                plt.show()
        else:
            raise Exception('El modelo no esta resuelto.')
    
    def set_initial_value(self, t0, y0):
        super().set_initial_value(t0, y0)
