import numpy as np
import matplotlib.pyplot as plt
try:
    import numba_models as nm
    from SODE_model import COMPARTMENTAL_Model
except:
    import models.numba_models as nm
    from models.SODE_model import COMPARTMENTAL_Model

class SIR_Model(COMPARTMENTAL_Model):

    def __init__(self) -> None:
        super().__init__(['Susceptibles', 'Infestados', 'Recuperados'],
                        ['S', 'I', 'R'], ['Beta', 'Gamma'],'SIR')
        
    def fun_SODE(self, t, y) -> np.ndarray:
        return nm.fun_SIR_SODE(t,y,
            beta= self._params[0],
            gamma= self._params[1],
        )
    
    def fun_args_SODE(self, t, y, args):
        return nm.fun_SIR_SODE(t,y,
            beta= args[0],
            gamma= args[1],
        )
    
    def plot_result(self, mode='all',
                    xlabel='Tiempo (dias)',
                    ylabel='Cantidad de personas (unidades)',
                    show=True,
                    grid=True):
        
        if self._result is not None:
            mode = mode.lower()
            t = self._result.t
            ys = self._result.y.reshape(self._total_compartmentals, t.shape[0])
            
            if mode == 'all':
                for i in range(self._total_compartmentals):
                    plt.plot(t, ys[i], label= self._compartmentals_name[i])
            elif (index := self._compartmentals_dict.get(mode)) is not None:
                plt.plot(t, ys[index].sum(axis=0), label= self._compartmentals_name[index])
            elif mode == 'umbral':
                u = self._params[1]/self._params[0]
                plt.axhline(u, color= 'black', ls='--', label='Umbral de Susceptibles')
            else:
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


class SIS_Model(COMPARTMENTAL_Model):

    def __init__(self) -> None:
        super().__init__(['Susceptibles', 'Infestados'],
                        ['S', 'I'], ['Beta', 'Gamma'],'SIS')
        
    def fun_SODE(self, t, y) -> np.ndarray:
        return nm.fun_SIS_SODE(t,y,
            beta= self._params[0],
            gamma= self._params[1],
        )
    
    def fun_args_SODE(self, t, y, args):
        return nm.fun_SIS_SODE(t,y,
            beta= args[0],
            gamma= args[1],
        )
    
    def plot_result(self, mode='all',
                    xlabel='Tiempo (dias)',
                    ylabel='Cantidad de personas (unidades)',
                    show=True,
                    grid=True):
        
        if self._result is not None:
            mode = mode.lower()
            t = self._result.t
            ys = self._result.y.reshape(self._total_compartmentals, t.shape[0])
            
            if mode == 'all':
                for i in range(self._total_compartmentals):
                    plt.plot(t, ys[i], label= self._compartmentals_name[i])
            elif (index := self._compartmentals_dict.get(mode)) is not None:
                plt.plot(t, ys[index].sum(axis=0), label= self._compartmentals_name[index])
            elif mode == 'umbral':
                u = self._params[1]/self._params[0]
                plt.axhline(u, color= 'black', ls='--', label='Umbral de Susceptibles')
            else:
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

class SEIR_Model(COMPARTMENTAL_Model):

    def __init__(self) -> None:
        super().__init__(['Susceptibles', 'Expuestos', 'Infestados', 'Recuperados'],
                        ['S', 'E', 'I', 'R'], ['Beta', 'Gamma', 'Sigma'],'SEIR')
        
    def fun_SODE(self, t, y) -> np.ndarray:
        return nm.fun_SEIR_SODE(t,y,
            beta= self._params[0],
            gamma= self._params[1],
            sigma= self._params[2],
        )
    
    def fun_args_SODE(self, t, y, args):
        return nm.fun_SEIR_SODE(t,y,
            beta= args[0],
            gamma= args[1],
            sigma= args[2],
        )
    
    def plot_result(self, mode='all',
                    xlabel='Tiempo (dias)',
                    ylabel='Cantidad de personas (unidades)',
                    show=True,
                    grid=True):
        
        if self._result is not None:
            mode = mode.lower()
            t = self._result.t
            ys = self._result.y.reshape(self._total_compartmentals, t.shape[0])
            
            if mode == 'all':
                for i in range(self._total_compartmentals):
                    plt.plot(t, ys[i], label= self._compartmentals_name[i])
            elif (index := self._compartmentals_dict.get(mode)) is not None:
                plt.plot(t, ys[index].sum(axis=0), label= self._compartmentals_name[index])
            elif mode == 'umbral':
                u = self._params[1]/self._params[0]
                plt.axhline(u, color= 'black', ls='--', label='Umbral de Susceptibles')
            else:
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

