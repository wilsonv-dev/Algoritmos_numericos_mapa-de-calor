"""
CAPA 4 - MÉTODO SSOR (Symmetric Successive Over-Relaxation)
Variante simétrica de SOR: realiza un barrido hacia adelante y luego uno hacia atrás
en cada iteración. Esto lo hace más estable y apropiado para matrices simétricas
como las que genera la ecuación de Laplace (nuestra malla de temperatura).
"""

import numpy as np
import time


def resolver(A, b, tolerancia=1e-6, max_iteraciones=1000, omega=1.25):
    """
    Resuelve el sistema lineal Ax = b usando el método SSOR.

    Cada iteración tiene DOS barridos:
        1) Barrido hacia adelante (i = 0, 1, ..., n-1): igual que SOR normal
        2) Barrido hacia atrás   (i = n-1, ..., 1, 0): SOR en orden inverso

    Esto garantiza simetría en el proceso de actualización.

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

        # --- BARRIDO HACIA ADELANTE (igual que SOR) ---
        for i in range(n):
            suma_inferior = sum(A[i, j] * x_actual[j] for j in range(i))
            suma_superior = sum(A[i, j] * x_anterior[j] for j in range(i + 1, n))
            valor_gs = (b[i] - suma_inferior - suma_superior) / A[i, i]
            x_actual[i] = (1 - omega) * x_anterior[i] + omega * valor_gs

        # guardamos el estado intermedio para el barrido hacia atrás
        x_medio = x_actual.copy()

        # --- BARRIDO HACIA ATRÁS (recorremos en orden inverso) ---
        for i in range(n - 1, -1, -1):
            suma_inferior = sum(A[i, j] * x_medio[j] for j in range(i))
            suma_superior = sum(A[i, j] * x_actual[j] for j in range(i + 1, n))
            valor_gs = (b[i] - suma_inferior - suma_superior) / A[i, i]
            x_actual[i] = (1 - omega) * x_medio[i] + omega * valor_gs

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