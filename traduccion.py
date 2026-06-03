"""
CAPA 3 - TRADUCCIÓN
===================
Esta capa es el puente entre el mundo visual (imagen PNG/JPG) y el mundo
matemático (sistema lineal Ax = b).

¿POR QUÉ ESTA CAPA EXISTE?
    Los métodos numéricos (Jacobi, Gauss-Seidel, SOR, SSOR) solo saben
    resolver sistemas de la forma Ax = b. No saben nada de imágenes.
    Esta capa convierte la imagen en ese sistema y luego convierte
    la solución de vuelta en una imagen.

¿CÓMO SE MODELA LA TEMPERATURA DESDE UNA IMAGEN?
    Un panel solar recibe irradiación solar desigual:
    - Zonas más iluminadas → píxeles más claros → temperatura más alta
    - Zonas en sombra      → píxeles más oscuros → temperatura más baja

    Cada píxel de la imagen se convierte en un nodo de una malla 2D.
    La temperatura en cada nodo debe satisfacer la ecuación de Laplace
    discreta (equilibrio térmico con sus vecinos):

        T(i-1,j) + T(i+1,j) + T(i,j-1) + T(i,j+1) - 4*T(i,j) = -Q(i,j)

    Donde Q(i,j) es el calor generado en ese nodo (derivado del brillo
    del píxel). Esto genera el sistema Ax = b que los métodos resolverán.

DEPENDENCIAS:
    - numpy  : operaciones matriciales
    - Pillow : lectura y redimensión de imágenes (pip install pillow)
"""

import numpy as np
from PIL import Image


# ─────────────────────────────────────────────
# CONSTANTES DE ESCALA TÉRMICA
# ─────────────────────────────────────────────
TEMP_MINIMA = 25.0    # °C — temperatura mínima (píxel completamente negro)
TEMP_MAXIMA = 80.0    # °C — temperatura máxima (píxel completamente blanco)
TEMP_RANGO  = TEMP_MAXIMA - TEMP_MINIMA   # = 55.0 °C de rango


def cargar_imagen_grises(ruta_imagen, tamanio_malla=20):
    """
    Lee una imagen del disco y la convierte a escala de grises
    redimensionada al tamaño de malla solicitado.

    ¿POR QUÉ REDIMENSIONAR?
        Una imagen de 1000×1000 píxeles generaría un sistema de
        1,000,000 ecuaciones — imposible de resolver en tiempo razonable
        con métodos iterativos. Con tamanio_malla=20 tenemos 400 nodos,
        que es manejable y visualmente claro.

    Parámetros:
        ruta_imagen  : ruta al archivo de imagen (JPG, PNG, BMP)
        tamanio_malla: cantidad de nodos por lado (ej. 20 → malla 20×20)

    Retorna:
        numpy array 2D de forma (tamanio_malla, tamanio_malla)
        con valores enteros 0–255 (escala de grises)
    """
    imagen = Image.open(ruta_imagen)

    # convertimos a escala de grises (modo 'L' en Pillow = Luminance)
    imagen_grises = imagen.convert('L')

    # redimensionamos con LANCZOS (mejor calidad al reducir tamaño)
    imagen_pequena = imagen_grises.resize(
        (tamanio_malla, tamanio_malla),
        Image.LANCZOS
    )

    # convertimos a array numpy con valores float para operar matemáticamente
    return np.array(imagen_pequena, dtype=float)


def pixeles_a_temperaturas(matriz_pixeles):
    """
    Convierte valores de píxeles (0–255) a temperaturas en °C.

    La relación es lineal:
        temperatura = TEMP_MINIMA + (pixel / 255) * TEMP_RANGO

    Ejemplo:
        pixel = 0   → temperatura = 25°C  (zona fría, en sombra)
        pixel = 128 → temperatura = 52°C  (zona media)
        pixel = 255 → temperatura = 80°C  (zona muy iluminada)

    ¿POR QUÉ ESTA CONVERSIÓN?
        Los paneles solares operan típicamente entre 25°C (ambiente)
        y 80°C (máximo bajo pleno sol). Mapear el brillo a temperatura
        es físicamente razonable: más luz → más calor.

    Parámetros:
        matriz_pixeles: numpy array 2D con valores 0–255

    Retorna:
        numpy array 2D con temperaturas en °C
    """
    return TEMP_MINIMA + (matriz_pixeles / 255.0) * TEMP_RANGO


def construir_sistema_laplace(matriz_temp, temp_borde=None):
    """
    Construye el sistema lineal Ax = b a partir de la malla de temperaturas.

    ¿QUÉ ES LA ECUACIÓN DE LAPLACE DISCRETA?
        En equilibrio térmico, la temperatura en un nodo interior es
        el promedio de sus 4 vecinos (arriba, abajo, izquierda, derecha):

            T_arriba + T_abajo + T_izq + T_der - 4*T_centro = -Q_centro

        En forma matricial esto se convierte en Ax = b donde:
            - A es la matriz de coeficientes (diagonal dominante ✓)
            - x es el vector de temperaturas desconocidas (nodos interiores)
            - b incorpora las condiciones de frontera (bordes del panel)

    ¿POR QUÉ DIAGONAL DOMINANTE?
        La ecuación de Laplace siempre produce una matriz diagonal dominante.
        Esto garantiza que Jacobi, Gauss-Seidel, SOR y SSOR CONVERGEN.
        Es la justificación matemática de por qué elegimos este problema.

    CONDICIONES DE FRONTERA (bordes del panel):
        Los bordes del panel intercambian calor con el ambiente.
        Si no se especifica temp_borde, usamos la temperatura promedio
        de los bordes de la imagen original.

    Parámetros:
        matriz_temp: numpy array 2D de temperaturas (NxN)
        temp_borde : temperatura fija en los bordes del panel (°C)
                     Si es None, se calcula como promedio de los bordes

    Retorna:
        A : numpy array 2D — matriz de coeficientes del sistema
        b : numpy array 1D — vector de términos independientes
        N : tamaño de la malla (para reconstruir la imagen después)
    """
    N = matriz_temp.shape[0]   # número de nodos por lado
    n_nodos = N * N            # total de nodos = N²

    # temperatura de borde: promedio de los 4 bordes de la imagen
    if temp_borde is None:
        borde_superior = matriz_temp[0, :]
        borde_inferior = matriz_temp[-1, :]
        borde_izq      = matriz_temp[:, 0]
        borde_der      = matriz_temp[:, -1]
        temp_borde = np.mean(np.concatenate([
            borde_superior, borde_inferior, borde_izq, borde_der
        ]))

    # inicializamos A (matriz cuadrada n_nodos × n_nodos) y b (vector n_nodos)
    A = np.zeros((n_nodos, n_nodos))
    b = np.zeros(n_nodos)

    def indice(i, j):
        """Convierte coordenadas (fila i, columna j) a índice lineal del vector x."""
        return i * N + j

    for i in range(N):
        for j in range(N):
            idx = indice(i, j)   # índice lineal de este nodo

            # ── DIAGONAL PRINCIPAL: coeficiente del nodo actual ──
            # La ecuación de Laplace tiene -4 * T_centro
            A[idx, idx] = -4.0

            # ── VECINO ARRIBA (i-1, j) ──
            if i > 0:
                A[idx, indice(i - 1, j)] = 1.0
            else:
                # borde superior: su temperatura ya es conocida → pasa a b
                b[idx] -= temp_borde

            # ── VECINO ABAJO (i+1, j) ──
            if i < N - 1:
                A[idx, indice(i + 1, j)] = 1.0
            else:
                # borde inferior
                b[idx] -= temp_borde

            # ── VECINO IZQUIERDA (i, j-1) ──
            if j > 0:
                A[idx, indice(i, j - 1)] = 1.0
            else:
                # borde izquierdo
                b[idx] -= temp_borde

            # ── VECINO DERECHA (i, j+1) ──
            if j < N - 1:
                A[idx, indice(i, j + 1)] = 1.0
            else:
                # borde derecho
                b[idx] -= temp_borde

            # ── FUENTE DE CALOR: la imagen dice cuánto calor genera este nodo ──
            # Restamos la temperatura original del píxel como término fuente
            b[idx] -= matriz_temp[i, j]

    return A, b, N


def solucion_a_mapa_calor(solucion, N):
    """
    Convierte el vector solución x (resultado de los métodos numéricos)
    de vuelta a una matriz 2D (NxN) de temperaturas.

    ¿POR QUÉ ES NECESARIO?
        Los métodos trabajan con vectores 1D. Para visualizar el mapa
        de calor necesitamos la forma matricial 2D original.

    Parámetros:
        solucion: numpy array 1D — vector x resultado del método
        N       : tamaño de la malla (lado de la matriz cuadrada)

    Retorna:
        numpy array 2D de forma (N, N) con las temperaturas resueltas
    """
    mapa = solucion.reshape((N, N))

    # recortamos valores fuera del rango físico esperado
    mapa = np.clip(mapa, TEMP_MINIMA - 10, TEMP_MAXIMA + 10)

    return mapa