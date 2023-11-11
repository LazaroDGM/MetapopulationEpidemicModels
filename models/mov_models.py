import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pydot
try:
    import numba_models as nm
    from SODE_model import SODE_Model
except:
    import models.numba_models as nm
    from models.SODE_model import SODE_Model

def hsv_to_hex(h, s, v=1):
    return hex(np.int32(matplotlib.colors.hsv_to_rgb([h,s,v])*255) @ np.array([16**4, 16**2, 1]))[2:]

class FLUX_Model(SODE_Model):

    def __init__(self) -> None:
        super().__init__([None], ['F'], 'Flux Model')

    def set_params(self, F: dict):
        try:
            self._params[0] = F
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
        
    def plot_result(self,mode='all',
                    xlabel='Tiempo (dias)',
                    ylabel='Cantidad de personas (unidades)',
                    show=True,
                    grid=True,
                    name_nodes= None):
        if self._result is not None:
            mode = mode.lower()
            t = self._result.t
            ys = self._result.y
            if name_nodes is None:
                name_nodes = [f'Personas en el Nodo {i+1}' for i in range(ys.shape[0])]
            if mode == 'all':
                for i, y in enumerate(ys):
                    plt.plot(t, ys[i], label= name_nodes[i])
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
        
    def generate_graph_population_pydot(self, name_nodes, y, edge_alpha=10, edge_len=3):
        G = pydot.Dot('G', graph_type='digraph', layout='circo')
        for i in range(len(name_nodes)):
            G.add_node(pydot.Node(name=name_nodes[i],
                                style='filled',
                                fillcolor=f'#{hsv_to_hex(240/360,y[i]/y.sum(),1)}'))

        for i in range(len(name_nodes)):
            for j in range(len(name_nodes)):
                if self._params[0][i,j] > 0:
                    G.add_edge(pydot.Edge(name_nodes[i],
                            name_nodes[j],
                            arrowsize=1,
                            len=edge_len,
                            color=f'#{hsv_to_hex(0,0,1-self._params[0][i,j]*edge_alpha)}'))
        return G
    
    def set_initial_value(self, t0, y0):
        super().set_initial_value(t0, y0)


def GenerateRandomArray(n):
    r = np.sort(np.random.random(n))
    a = np.zeros(n)
    a[:n-1] = r[1:n]
    a[n-1] = 1
    return a - r

def GenerateRandomMatrix(n, m, max= None):    
    if max is None:
        max = np.random.random(n)

    r = np.zeros((n,m))
    for i in range(n):
        r[i] = np.sort(np.random.uniform(0,max[i], m))
    a = np.zeros((n,m))
    a[:,:m-1] = r[:,1:m]
    a[:,m-1] = max
    return a - r

def GenerateParams_EULER(K):
    F = GenerateRandomMatrix(K,K)
    np.fill_diagonal(F,0)
    return F

def GenerateY0_EULER(K):
    y0 = np.round(np.random.rand(K) * 5000 + 1000, 0)
    return y0

def GenerateParams_SIR_LAGRANGE(K, in_mode= 'rand', a= 1):

    Out = GenerateRandomMatrix(K,K)
    np.fill_diagonal(Out,0)

    if in_mode == 'alpha-rand':
        In = np.random.random() * Out
    elif in_mode == 'alpha-const':
        In = a * Out
    elif in_mode == 'rand':
        In = np.random.random((K,K))
        np.fill_diagonal(In,0)
    elif in_mode == 'const':
        In = np.zeros((K,K)) + a
    else:
        raise Exception('Modo incorrecto')
        
    Beta = np.random.rand(K)
    Gamma = np.random.rand(K)

    return Out, In, Beta, Gamma

def GenerateY0_SIR_LAGRANGE(K):
    y0 = np.zeros((4,K,K))
    np.fill_diagonal(y0[3], np.round(np.random.rand(K) * 5000 + 1000, 0))
    np.fill_diagonal(y0[1], np.round(np.random.rand(K) * 5 + 1))
    y0[0] = y0[3] - y0[1]
    y0.resize((4*K*K,))
    return y0
