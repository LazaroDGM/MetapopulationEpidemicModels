# Experimento 02

---
En este pipeline, se prueba la eficiencia de 4 métodos diferentes que implementan el sistema de ecuaciones diferenciales de un modelo SIR con movimiento Lagrangiano para su evaluación en el algoritmo de resolución numérica `solve_ivp` de `scipy.integrate`. Se crearon $19$ conjuntos de parámetros con $30$ versiones de parámetros diferentes generados aleatoriamente. Cada conjunto de parámetros se confeccionó atendiendo la cantidad de nodos de la red del modelo matemático, en este caso siendo $K$ la cantidad de nodos, se generaron conjuntos de parámetros para $K = 2,3,...,20$.

Los **casos 1 y 2** son implementaciones comunes en `Python`. En una la función principal crea dinámicamente una función en términos de $t$ y $y$, pero que ya tiene implícito los parámetros del modelo y es la que se usa para evaluar el SEDO. En cambio la otra, está en términos de $t$, $y$ y el resto de parámetros, donde se evalúa directamente el SEDO.

Los **casos 3 y 4** son implementaciones en `Python` que incorporan la compilación `JIT`, por medio del módulo `numba`. En estos casos las funciones se compilarán una vez creadas y luego, cada vez que sean usadas NO serán interpretadas, como normalemnte trabaja Python. Apartando este elemento, ambas implementaciones son similares a los **casos 1 y 2**.

## Generación de Parámetros

La generación de parámetros se realizó para cada valor $K$ de nodos de la red del modelo de movimiento. En este caso solo se probaron con valores de $K = 2,3,...,20$. Para cada nodo se generaron $30$ casos de parámetros. Para los parámetros se generaba tanto las matrices $\phi$ y $\tau$ de dimensiones $(K \times K)$ con valores reales en el intervalo $(0,1)$ y diagonal nula; como los vectores $\beta$ y $\gamma$ de tamaño $K$, con valores nuevamente en el intervalo $(0,1)$.

Una vez generados estos conjuntos de datos fueron guardados en un objeto comprimido de `numpy` para ser reutilizados en otro momento, sin tener que generar nuevamente los parámetros. Además resolver estos SEDO, se necesitan valores iniciales. En este caso se utilizaron los mismos valores iniciales para cada nodo, teniendo en cuenta la cantidad de nodos. Estos valores fueron los siguientes:

- $S_i(0) = 1995$
- $I_i(0) = 5$
- $R_i(0) = 0$
- $N_i(0) = 2000$

Y por último, el intervalo de estudio y resolución de los SEDOs se seleccionó en el intervalo $t \in (0,500)$.

## Simulaciones

Para cada cantidad total de nodos en la red, se realizan $30$ simulaciones (una por cada juego de parámetros). De cada simulación se registra el tiempo que demora todo la ejecución que implica resolver un SEDO con `solve_ivp` usando las funciones implementadas. Por cada **Método** se realizan todas estas simulaciones y los resultados se guardan en archivos comprimidos generados por `numpy` para reutilizarlos sin necesidad de realizar nueva mente las simulaciones.

Al finalizar las simulaciones se ilustran las gráficas de tiempo promedio que tomó el experimento para cada cantidad total de nodos en la red, es decir, la complejidad temporal que toma la resolución de un SEDO. Además se hace una regresión de 2do orden con estos valores y se ilustra en una segunda gráfica la estimación de la complejidad temporal del método para valores mayores de la cantidad de nodos empleados.

Hacer este tipo de experimentos nos dará una idea de cuán eficiente será la estimación de parámetros en un futuro al utilizar el método con mejor rendimiento. Además por cada juego de parámetros solo se hace una única resolución numérica en el experimento, ya que los algoritmos que se emplean para la estimación de parámetros evaluan una sola vez cada juego de parámetros que utilizan.

### Método 1

En este caso `gen_fun_sir_lagrange` recibe los parámetros del modelo y genera una función para evaluar el modelo con estos parámetros ya insertados. Por lo tanto una vez generada la nueva función esta es la que se debe utilizar para buscar una solución numérica al SEDO.

![Método 1 Real](/experiments/experiment_02/img/method_1_real.png)

![Método 1 Estimación](/experiments/experiment_02/img/method_1_estimation.png)

### Método 2

En este caso `fun_sir_lagrange` recibe tanto los parámetros del modelo, como puede ser usado para evaluarlo. Por lo tanto esta función se debe utilizar directamente para buscar una solución numérica al SEDO.

![Método 2 Real](/experiments/experiment_02/img/method_2_real.png)

![Método 2 Estimación](/experiments/experiment_02/img/method_2_estimation.png)

### Método 3

Similar al **Método 1** donde ahora `gen_fun_sir_lagrange_numba` recibe los parámetros del modelo y genera una función para evaluar el modelo con estos parámetros ya insertados, pero esta función generada se decora con `njit` forzando que se compile en primera instancia. Por lo tanto una vez generada la nueva función esta es la que se debe utilizar para buscar una solución numérica al SEDO.

![Método 3 Real](/experiments/experiment_02/img/method_3_real.png)

![Método 3 Estimación](/experiments/experiment_02/img/method_3_estimation.png)

### Método 4

Similar al **Método 2** donde en este caso `fun_sir_lagrange_numba` recibe tanto los parámetros del modelo, como puede ser usado para evaluarlo, pero se utiliza `njit` para decorarla y trabajar con una versión compilada de esta. Por lo tanto esta función se debe utilizar directamente para buscar una solución numérica al SEDO.

![Método 4 Real](/experiments/experiment_02/img/method_4_real.png)

![Método 4 Estimación](/experiments/experiment_02/img/method_4_estimation.png)

## Resumen

Para finalizar ilustramos todas las curvas de complejidad temporal real simuladas, para facilitar la comparación de estas. Así destacamos, la superioridad de los **Métodos 3 y 4** que utilizan la compilación `JIT`. Mientras que para los **Métodos 1 y 2** aunque la curva de creciemiento es más acelerada, ambas no tienen mucha diferencias entre ellas. Sin embargo algo que llama mucho la atención es el tiempo adicional que toma el **Método 3** con respecto al **Método 4**, donde para generar dinámicamente la función que deseamos compilar consume un tiempo extra, que empeora mucho la complejidad cuando se compara con el último Método.

![Comparación Final](/experiments/experiment_02/img/final_results.png)