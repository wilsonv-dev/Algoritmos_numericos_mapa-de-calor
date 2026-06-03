"""
CAPA 2 - CONTROLADOR
====================
El controlador es el director de orquesta del proyecto. No calcula nada
por sí mismo: su único trabajo es coordinar las otras capas en el orden
correcto y validar que los datos del usuario sean correctos antes de
pasarlos a los métodos.

FLUJO DE TRABAJO QUE COORDINA:
    1. Recibe los parámetros del usuario (desde la interfaz)
    2. Valida que los parámetros sean correctos
    3. Llama a Traducción (Capa 3) para construir el sistema Ax = b
    4. Llama a los Métodos (Capa 4) seleccionados por el usuario
    5. Llama a Servicios (Capa 5) para graficar y exportar
    6. Devuelve los resultados a la Interfaz (Capa 1)

¿POR QUÉ ESTA CAPA ES IMPORTANTE?
    Sin controlador, la interfaz tendría que saber cómo llamar a los
    métodos y cómo construir el sistema — mezclando responsabilidades.
    El controlador mantiene cada capa ignorante de las demás.
"""

import numpy as np

# importamos las otras capas
import traduccion
import servicios
from metodos_numericos import jacobi, gauss_seidel, sor, ssor


# ─────────────────────────────────────────────
# CONSTANTES DE VALIDACIÓN
# ─────────────────────────────────────────────
OMEGA_MIN        = 0.01    # mínimo valor permitido para omega
OMEGA_MAX        = 1.99    # máximo valor permitido para omega
TOLERANCIA_MIN   = 1e-12   # tolerancia mínima permitida
TOLERANCIA_MAX   = 1.0     # tolerancia máxima permitida (sería impreciso)
ITERACIONES_MIN  = 10      # mínimo de iteraciones
ITERACIONES_MAX  = 5000    # máximo de iteraciones (evita colgarse)
MALLA_MIN        = 5       # malla mínima (5×5)
MALLA_MAX        = 50      # malla máxima (50×50 = 2500 nodos, manejable)


def validar_parametros(tolerancia, max_iteraciones, omega, tamanio_malla):
    """
    Verifica que los parámetros ingresados por el usuario sean válidos.

    ¿POR QUÉ VALIDAR AQUÍ Y NO EN LA INTERFAZ?
        La interfaz puede tener validación visual (campos en rojo), pero
        la validación de lógica de negocio debe estar en el controlador.
        Así si alguien usa el controlador desde otro contexto, sigue seguro.

    Parámetros:
        tolerancia      : float — criterio de parada
        max_iteraciones : int   — límite de iteraciones
        omega           : float — factor de relajación para SOR/SSOR
        tamanio_malla   : int   — tamaño de la malla NxN

    Retorna:
        lista de strings con los errores encontrados (vacía si todo es válido)
    """
    errores = []

    # validar tolerancia
    if not (TOLERANCIA_MIN <= tolerancia <= TOLERANCIA_MAX):
        errores.append(
            f"Tolerancia debe estar entre {TOLERANCIA_MIN} y {TOLERANCIA_MAX}. "
            f"Valor recibido: {tolerancia}"
        )

    # validar iteraciones
    if not (ITERACIONES_MIN <= int(max_iteraciones) <= ITERACIONES_MAX):
        errores.append(
            f"Iteraciones debe estar entre {ITERACIONES_MIN} y {ITERACIONES_MAX}. "
            f"Valor recibido: {max_iteraciones}"
        )

    # validar omega (solo aplica a SOR y SSOR, pero lo validamos siempre)
    if not (OMEGA_MIN <= omega <= OMEGA_MAX):
        errores.append(
            f"Omega (ω) debe estar en el intervalo ({OMEGA_MIN}, {OMEGA_MAX}). "
            f"Valor recibido: {omega}"
        )

    # validar tamaño de malla
    if not (MALLA_MIN <= int(tamanio_malla) <= MALLA_MAX):
        errores.append(
            f"Tamaño de malla debe estar entre {MALLA_MIN} y {MALLA_MAX}. "
            f"Valor recibido: {tamanio_malla}"
        )

    return errores


def ejecutar_simulacion(ruta_imagen, metodos_seleccionados,
                        tolerancia=1e-6, max_iteraciones=500,
                        omega=1.25, tamanio_malla=20):
    """
    Función principal del controlador. Orquesta todo el proceso de simulación.

    PASO A PASO:
        1. Valida los parámetros
        2. Carga y convierte la imagen a malla de temperaturas (Capa 3)
        3. Construye el sistema Ax = b (Capa 3)
        4. Ejecuta cada método seleccionado (Capa 4)
        5. Convierte las soluciones a mapas de calor (Capa 3)
        6. Devuelve todo a la interfaz (Capa 1)

    Parámetros:
        ruta_imagen           : str  — ruta al archivo de imagen
        metodos_seleccionados : list — lista con nombres de métodos a ejecutar
                                       Ej: ["Jacobi", "SOR"]
        tolerancia            : float
        max_iteraciones       : int
        omega                 : float — solo usado por SOR y SSOR
        tamanio_malla         : int   — resolución de la malla NxN

    Retorna:
        dict con:
            - "error"          : mensaje de error (None si todo salió bien)
            - "mapa_original"  : mapa de temperaturas de la imagen original
            - "resultados"     : dict nombre_metodo → resultado numérico
            - "mapas_calor"    : dict nombre_metodo → mapa 2D de temperaturas
            - "A"              : matriz del sistema (para referencia)
            - "b"              : vector del sistema (para referencia)
            - "N"              : tamaño de la malla
    """

    # ── PASO 1: Validar parámetros ──
    errores = validar_parametros(tolerancia, max_iteraciones, omega, tamanio_malla)
    if errores:
        return {"error": "\n".join(errores)}

    if not metodos_seleccionados:
        return {"error": "Debes seleccionar al menos un método numérico."}

    # ── PASO 2: Cargar imagen y convertir a malla de temperaturas ──
    try:
        matriz_pixeles = traduccion.cargar_imagen_grises(ruta_imagen, tamanio_malla)
        mapa_original  = traduccion.pixeles_a_temperaturas(matriz_pixeles)
    except FileNotFoundError:
        return {"error": f"No se encontró la imagen en: {ruta_imagen}"}
    except Exception as e:
        return {"error": f"Error al cargar la imagen: {str(e)}"}

    # ── PASO 3: Construir el sistema lineal Ax = b ──
    try:
        A, b, N = traduccion.construir_sistema_laplace(mapa_original)
    except Exception as e:
        return {"error": f"Error al construir el sistema lineal: {str(e)}"}

    # ── PASO 4: Ejecutar los métodos seleccionados ──
    resultados    = {}   # nombre → resultado numérico del método
    mapas_calor   = {}   # nombre → mapa 2D de temperaturas resuelto

    # mapeo de nombres a módulos de métodos
    metodos_disponibles = {
        "Jacobi":       jacobi,
        "Gauss-Seidel": gauss_seidel,
        "SOR":          sor,
        "SSOR":         ssor,
    }

    for nombre in metodos_seleccionados:
        if nombre not in metodos_disponibles:
            continue

        modulo = metodos_disponibles[nombre]

        try:
            # SOR y SSOR necesitan el parámetro omega
            if nombre in ("SOR", "SSOR"):
                resultado = modulo.resolver(
                    A, b,
                    tolerancia=tolerancia,
                    max_iteraciones=int(max_iteraciones),
                    omega=omega
                )
            else:
                resultado = modulo.resolver(
                    A, b,
                    tolerancia=tolerancia,
                    max_iteraciones=int(max_iteraciones)
                )

            resultados[nombre] = resultado

            # convertimos el vector solución a mapa 2D para visualizar
            mapa = traduccion.solucion_a_mapa_calor(resultado["solucion"], N)
            mapas_calor[nombre] = mapa

        except Exception as e:
            # si un método falla, lo registramos pero continuamos con los demás
            resultados[nombre] = {"error": str(e)}

    return {
        "error":         None,
        "mapa_original": mapa_original,
        "resultados":    resultados,
        "mapas_calor":   mapas_calor,
        "A":             A,
        "b":             b,
        "N":             N
    }