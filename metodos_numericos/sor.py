"""
CAPA 4 - MÉTODO SOR (Successive Over-Relaxation)
Extensión de Gauss-Seidel con un factor de relajación omega (w).
Si w = 1 → equivale exactamente a Gauss-Seidel.
Si 1 < w < 2 → sobre-relajación (converge más rápido en muchos casos).
Si 0 < w < 1 → sub-relajación (útil para sistemas difíciles).
"""

import numpy as np
import time


def resolver(A, b, tolerancia=1e-6, max_iteraciones=1000, omega=1.25):
    """
    Resuelve el sistema lineal Ax = b usando el método SOR.

    Fórmula:
        x_i^(k+1) = (1 - w) * x_i^(k)
                  + (w / a_ii) * (b_i
                    - suma(a_ij * x_j^(k+1)) para j < i
                    - suma(a_ij * x_j^(k))   para j > i)

    Parámetros:
        A             : matriz de coeficientes (numpy array n x n)
        b             : vector de términos independientes (numpy array n)
        tolerancia    : criterio de parada por error relativo
        max_iteraciones: número máximo de iteraciones permitidas
        omega         : factor de relajación (debe estar en el intervalo (0, 2))

    Retorna:
        dict con:
            - solucion      : vector x final
            - iteraciones   : número de iteraciones realizadas
            - error_final   : error en la última iteración
            - historial_error: lista de errores por iteración
            - tiempo        : tiempo de cómputo en segundos
            - convergio     : True si el método convergió
    """

    if not (0 < omega < 2):
        raise ValueError(f"omega debe estar en (0, 2). Valor recibido: {omega}")

    n = len(b)
    x_actual = np.zeros(n)
    historial_error = []
    error = float('inf')
    iteracion = 0

    tiempo_inicio = time.time()

    while error > tolerancia and iteracion < max_iteraciones:

        x_anterior = x_actual.copy()

        for i in range(n):
            # parte de Gauss-Seidel (sin omega)
            suma_inferior = sum(A[i, j] * x_actual[j] for j in range(i))
            suma_superior = sum(A[i, j] * x_anterior[j] for j in range(i + 1, n))

            valor_gs = (b[i] - suma_inferior - suma_superior) / A[i, i]

            # aplicamos el factor de relajación omega
            x_actual[i] = (1 - omega) * x_anterior[i] + omega * valor_gs

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
        "convergio": error <= tolerancia,
        "omega_usado": omega
    }