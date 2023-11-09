from scipy.integrate import solve_ivp
import numpy as np
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error
try:
    from clasic_epidemic_models import SIR_Model
    from metapopulation_epidemic_model import SIR_EULER_Model, SIR_LAGRANGE_Model
except:
    from models.clasic_epidemic_models import SIR_Model
    from models.metapopulation_epidemic_model import SIR_EULER_Model, SIR_LAGRANGE_Model

#result = differential_evolution(fitness_DE, [(0, 1), (0, 1)],
#                                maxiter=100,disp=False,polish=False,
#                                workers=1, callback=lambda xk, convergence: fitness_DE(xk) < 0.1)

class DE_ESTIMATOR_SIR(SIR_Model):


    def __init__(self) -> None:
        super().__init__()

    def extract_info(self, result):
        y = result['y']
        S_t = y[0]
        I_t = y[1]
        C_t = self._params[0] * S_t * I_t
        return C_t

    def estimate_with_model(self, n):
        
        t_eval = np.linspace(0,n,n+1)
        self.solve(t_span=(0,n),t_eval=t_eval)
        original_result = self.get_resut()
        original_C = self.extract_info(original_result)

        def fitness(value):
            args = np.zeros((3,))
            args[:2] = value
            args[2] = self._params[2]
            estimate_result = self.solve_args(
                args=args,
                t_span=(0,n),
                t_eval=t_eval
                )
            estimate_C = self.extract_info(estimate_result)
            return mean_squared_error(original_C, estimate_C)
        
        result = differential_evolution(
            func= fitness,
            bounds= [(0,1), (0,1)],
            maxiter=100,
            disp=False,
            polish=True,
            workers=1,
            callback=lambda xk, convergence: fitness(xk) < 0.1
        )
        print(result)

class DE_ESTIMATOR_SIR_EULER(SIR_EULER_Model):


    def __init__(self, K) -> None:
        super().__init__(K)

    def extract_info(self, result, Beta):
        y = result['y'].reshape((4,self._K, result['t'].shape[0]))
        S_t = y[0]
        I_t = y[1]
        N_t = y[3]

        for i in range(self._K):
            S_t[i] = Beta[i] * S_t[i]
        C_t = I_t * S_t / N_t
        return C_t

    def estimate_with_model(self, n):
        
        t_eval = np.linspace(0,n,n+1)
        self.solve(t_span=(0,n),t_eval=t_eval)
        original_result = self.get_resut()
        original_C = self.extract_info(original_result, self._params[1])

        def fitness(value):
            args = [self._params[0], value, self._params[2]]
            estimate_result = self.solve_args(
                args=args,
                t_span=(0,n),
                t_eval=t_eval
                )
            estimate_C = self.extract_info(estimate_result, value)
            return mean_squared_error(original_C, estimate_C)
        
        result = differential_evolution(
            func= fitness,
            bounds= [(0,1)] * self._K,
            maxiter=100,
            disp=True,
            polish=True,
            workers=1,
            callback=lambda xk, convergence: fitness(xk) < 0.1
        )
        return result

class DE_ESTIMATOR_SIR_LAGRANGE(SIR_LAGRANGE_Model):


    def __init__(self, K) -> None:
        super().__init__(K)

    def extract_info(self, result, Beta):
        y = result['y'].reshape((4, self._K, self._K, result['t'].shape[0]))

        S_k_i = y[0].sum(axis=0)
        I_k_i = y[1].sum(axis=0)
        N_k_i = y[3].sum(axis=0)

        C_i_t = np.zeros((self._K,result['t'].shape[0]))
        for i in range(self._K):
            C_i_t[i] = Beta[i] * S_k_i[i] * I_k_i[i] / N_k_i[i]

        return C_i_t

    def estimate_with_model(self, n):
        
        t_eval = np.linspace(0,n,n+1)
        self.solve(t_span=(0,n),t_eval=t_eval)
        original_result = self.get_resut()
        original_C = self.extract_info(original_result, self._params[2])

        def fitness(value):
            args = [self._params[0], self._params[1], value, self._params[3]]
            estimate_result = self.solve_args(
                args=args,
                t_span=(0,n),
                t_eval=t_eval
                )
            estimate_C = self.extract_info(estimate_result, value)
            return mean_squared_error(original_C, estimate_C)
        
        result = differential_evolution(
            func= fitness,
            bounds= [(0,1)] * self._K,
            maxiter=100,
            disp=True,
            polish=True,
            workers=1,
            callback=lambda xk, convergence: fitness(xk) < 0.1
        )
        return result

        
    
#es = ESTIMATOR_SIR()
#es.set_params([0.2,0.01,1000])
#es.set_initial_value(0,[999,1,0])
#es.solve((0,300), np.linspace(0, 300, 61))
##es.plot_result()
#es.estimate_with_model(300)
##print(es.estimate_with_model()[0])

#es = ESTIMATOR_SIR_EULER(3)
#es.set_params([
#    np.array([
#        [0, 0.1, 0.2],
#        [0.2, 0, 0.3],
#        [0.13, 0.14, 0]
#    ]),
#    np.array([0.1,0.3,0.2]),
#    np.array([0.14,0.14,0.14]),
#])
#es.set_initial_value(0,
#    np.array(
#        [
#            [999,1999,1499],
#            [1,1,1],
#            [0,0,0],
#            [1000,2000,1500]
#        ]
#    ).flatten()
#    )
#es.solve((0,300))
##es.plot_result()
#    
#es.estimate_with_model(300)
