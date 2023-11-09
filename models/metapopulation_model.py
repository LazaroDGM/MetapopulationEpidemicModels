import matplotlib.pyplot as plt
import regex as re
import numpy as np
try:
    from SODE_model import COMPARTMENTAL_Model
except:
    from models.SODE_model import COMPARTMENTAL_Model

RE_COMPARTMENT_ALL = re.compile('(\w+)\-all')
RE_COMPARTMENT_NODE = re.compile('(\w+)\-(\d+)')

class EULER_COMPARTIMENTAL_Model(COMPARTMENTAL_Model):
    
    def __init__(self, compartmentals_name, compartmentals_short_name, params_name, K, name) -> None:
        super().__init__(compartmentals_name, compartmentals_short_name, ['F'] + params_name, name)

        if K < 2:
            raise Exception('La cantidad de nodos debe ser un entero mayor que 1')
        self._K = K

    def set_params(self, params: dict | list):
        if params is dict:
            if not (self._K == params['F'].shape[0] == params['F'].shape[1]):
                raise Exception(f'La matriz "F" debe ser de dimension ({self._K, self._K}).')
        else:
            if not (self._K == params[0].shape[0] == params[0].shape[1]):
                raise Exception(f'La matriz "F" debe ser de dimension ({self._K, self._K}).')
        super().set_params(params)
        

    def plot_result(self, mode='all',
                    xlabel='Tiempo (dias)',
                    ylabel='Cantidad de personas (unidades)',
                    show=True,
                    grid=True):
        
        if self._result is not None:
            mode = mode.lower()
            t = self._result.t
            ys = self._result.y[:self._total_compartmentals * self._K] \
                .reshape((self._total_compartmentals, self._K, t.shape[0]))
            
            if mode == 'all':
                for i in range(self._total_compartmentals):
                    plt.plot(t, ys[i].sum(axis=0), label= self._compartmentals_name[i])
            elif (index := self._compartmentals_dict.get(mode)) is not None:
                plt.plot(t, ys[index].sum(axis=0), label= self._compartmentals_name[index])
            elif r := RE_COMPARTMENT_ALL.fullmatch(mode):
                compartment_k = self._compartmentals_dict[r.group(1)]
                for i in range(self._K):
                    plt.plot(t, ys[compartment_k,i],
                            label= self._compartmentals_name[compartment_k] + f' {i+1}')
            elif r := RE_COMPARTMENT_NODE.fullmatch(mode):
                compartment_k = self._compartmentals_dict[r.group(1)]
                node = int(r.group(2))
                plt.plot(t, ys[compartment_k,node-1],
                        label= self._compartmentals_name[compartment_k] + f' {node}')
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

class LAGRANGE_COMPARTIMENTAL_Model(COMPARTMENTAL_Model):
    
    def __init__(self, compartmentals_name, compartmentals_short_name, params_name, K, name) -> None:
        super().__init__(compartmentals_name, compartmentals_short_name, ['Out', 'In'] + params_name, name)

        if K < 2:
            raise Exception('La cantidad de nodos debe ser un entero mayor que 1')
        self._K = K

    def set_params(self, params: dict | list):
        if params is dict:
            if not (self._K == params['Out'].shape[0] == params['Out'].shape[1]):
                raise Exception(f'La matriz "Out" debe ser de dimension ({self._K, self._K}).')
            if not (self._K == params['In'].shape[0] == params['In'].shape[1]):
                raise Exception(f'La matriz "In" debe ser de dimension ({self._K, self._K}).')
        else:
            if not (self._K == params[0].shape[0] == params[0].shape[1]):
                raise Exception(f'La matriz "Out" debe ser de dimension ({self._K, self._K}).')
            if not (self._K == params[1].shape[0] == params[1].shape[1]):
                raise Exception(f'La matriz "In" debe ser de dimension ({self._K, self._K}).')
        super().set_params(params)
        

    def plot_result(self,
                    mode='all',
                    xlabel='Tiempo (dias)',
                    ylabel='Cantidad de personas (unidades)',
                    show=True,
                    grid=True):
        
        if self._result is not None:
            mode = mode.lower()
            t = self._result.t
            ys = self._result.y[:self._total_compartmentals * self._K * self._K] \
                .reshape((self._total_compartmentals, self._K, self._K, t.shape[0]))
            
            if mode == 'all':
                for i in range(self._total_compartmentals):
                    plt.plot(t, ys[i].sum(axis=0).sum(axis=0), label= self._compartmentals_name[i])
            elif (index := self._compartmentals_dict.get(mode)) is not None:
                plt.plot(t, ys[index].sum(axis=0).sum(axis=0), label= self._compartmentals_name[index])
            elif r := RE_COMPARTMENT_ALL.fullmatch(mode):
                compartment_k = self._compartmentals_dict[r.group(1)]
                for i, y in enumerate(ys[compartment_k].sum(axis=0)):
                    plt.plot(t, y,
                            label= self._compartmentals_name[compartment_k] + f' {i+1}')
            elif r := RE_COMPARTMENT_NODE.fullmatch(mode):
                compartment_k = self._compartmentals_dict[r.group(1)]
                node = int(r.group(2))
                plt.plot(t, ys[compartment_k].sum(axis=0)[node-1],
                        label= self._compartmentals_name[compartment_k] + f' {node}')
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

