"""
CAPA 4 - MÉTODO DE JACOBI
Implementación pura del método iterativo de Jacobi para resolver Ax = b.
No contiene interfaz, ni gráficas, ni validaciones de usuario.
"""

import numpy as np
import time


def resolver(A, b, tolerancia=1e-6, max_iteraciones=1000):
    """
    Resuelve el sistema lineal Ax = b usando el método de Jacobi.

    Fórmula:
        x_i^(k+1) = (1 / a_ii) * (b_i - suma(a_ij * x_j^(k)) para j != i)

    Parámetros:
        A             : matriz de coeficientes (numpy array n x n)
        b             : vector de términos independientes (numpy array n)
        tolerancia    : criterio de parada por error relativo
        max_iteraciones: número máximo de iteraciones permitidas

    Retorna:
        dict con:
            - solucion      : vector x final
            - iteraciones   : número de iteraciones realizadas
            - error_final   : error en la última iteración
            - historial_error: lista de errores por iteración
            - tiempo        : tiempo de cómputo en segundos
            - convergio     : True si el método convergió
    """

    n = len(b)
    x_actual = np.zeros(n)          # vector solución inicial (todo ceros)
    x_nuevo = np.zeros(n)           # vector solución de la iteración siguiente
    historial_error = []            # guardamos el error de cada iteración
    error = float('inf')
    iteracion = 0

    tiempo_inicio = time.time()

    while error > tolerancia and iteracion < max_iteraciones:

        for i in range(n):
            # suma de a_ij * x_j para todos los j distintos de i
            suma = 0.0
            for j in range(n):
                if j != i:
                    suma += A[i, j] * x_actual[j]

            # fórmula de Jacobi
            x_nuevo[i] = (b[i] - suma) / A[i, i]

        # calculamos el error como la norma de la diferencia
        error = np.linalg.norm(x_nuevo - x_actual) / (np.linalg.norm(x_nuevo) + 1e-12)
        historial_error.append(error)

        # actualizamos x para la siguiente iteración (Jacobi usa x_actual completo)
        x_actual = x_nuevo.copy()
        iteracion += 1

    tiempo_total = time.time() - tiempo_inicio

    return {
        "solucion": x_actual,
        "iteraciones": iteracion,
        "error_final": error,
        "historial_error": historial_error,
        "tiempo": tiempo_total,
        "convergio": error <= tolerancia
    }