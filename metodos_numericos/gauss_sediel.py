"""
CAPA 4 - MÉTODO DE GAUSS-SEIDEL
Implementación pura del método iterativo de Gauss-Seidel para resolver Ax = b.
Diferencia clave con Jacobi: usa los valores ya actualizados en la misma iteración.
"""

import numpy as np
import time


def resolver(A, b, tolerancia=1e-6, max_iteraciones=1000):
    """
    Resuelve el sistema lineal Ax = b usando el método de Gauss-Seidel.

    Fórmula:
        x_i^(k+1) = (1 / a_ii) * (b_i
                    - suma(a_ij * x_j^(k+1)) para j < i   <- ya actualizados
                    - suma(a_ij * x_j^(k))   para j > i   <- aún sin actualizar)

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
    x_actual = np.zeros(n)          # vector solución inicial
    historial_error = []
    error = float('inf')
    iteracion = 0

    tiempo_inicio = time.time()

    while error > tolerancia and iteracion < max_iteraciones:

        x_anterior = x_actual.copy()   # guardamos para calcular el error al final

        for i in range(n):
            # suma con j < i: usa valores ya actualizados en esta iteración
            suma_inferior = sum(A[i, j] * x_actual[j] for j in range(i))

            # suma con j > i: usa valores de la iteración anterior
            suma_superior = sum(A[i, j] * x_anterior[j] for j in range(i + 1, n))

            # fórmula de Gauss-Seidel
            x_actual[i] = (b[i] - suma_inferior - suma_superior) / A[i, i]

        # error relativo entre iteración actual y anterior
        error = np.linalg.norm(x_actual - x_anterior) / (np.linalg.norm(x_actual) + 1e-12)
        historial_error.append(error)
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