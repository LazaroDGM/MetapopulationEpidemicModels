from SODE_model import SODE_Model
import numpy as np
import numba_models as nm
import matplotlib.pyplot as plt

class FLUX_Model(SODE_Model):

    def __init__(self) -> None:
        super().__init__()
        self._params = [None]
        self._params_name = ['F']
        self._params_correct = False

        self._name_model = 'Flux Model'

    def set_params(self, params: dict):
        try:
            if params is dict:
                self._params[0] = params['F']
            else:
                self._params[0] = params[0]
            self._params_correct = True
        except Exception as e:
            self._params_correct = False
            raise e
        
    def fun_SODE(self, t, y) -> np.ndarray:
        if self._params_correct:
            return nm.fun_EULER_SODE(t,y,
                F= self._params[0]
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

