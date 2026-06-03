"""
CAPA 5 - SERVICIOS
==================
Esta capa provee funciones de soporte que no pertenecen a ninguna otra capa:
    - Graficar la convergencia de los métodos
    - Exportar resultados a CSV o TXT
    - Guardar el mapa de calor como imagen PNG
    - Generar la figura comparativa de los 4 métodos

¿POR QUÉ UNA CAPA SEPARADA?
    Graficar y exportar no son responsabilidades de los métodos numéricos
    (que solo calculan) ni de la interfaz (que solo muestra en pantalla).
    Separarlo aquí permite reutilizar estos servicios desde cualquier capa
    sin crear dependencias cruzadas.

DEPENDENCIAS:
    - matplotlib : gráficas y mapas de calor
    - numpy      : operaciones con arrays
    - csv        : escritura de archivos CSV (incluido en Python estándar)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import csv
import os
from datetime import datetime


# ─────────────────────────────────────────────
# COLORES Y ESTILO
# ─────────────────────────────────────────────
# Colores para distinguir cada método en las gráficas
COLORES_METODOS = {
    "Jacobi":       "#E74C3C",   # rojo
    "Gauss-Seidel": "#3498DB",   # azul
    "SOR":          "#2ECC71",   # verde
    "SSOR":         "#F39C12",   # naranja
}


def graficar_convergencia(resultados_metodos, guardar_ruta=None):
    """
    Genera una gráfica de Error vs Iteraciones para uno o varios métodos.

    ¿QUÉ MUESTRA ESTA GRÁFICA?
        Muestra qué tan rápido cada método reduce el error en cada iteración.
        Una curva que baja más rápido = método que converge más rápido.
        El eje Y usa escala logarítmica porque los errores decrecen
        exponencialmente (de 1.0 a 0.000001 en pocas iteraciones).

    Parámetros:
        resultados_metodos: dict con nombre_metodo → resultado del método
                           Ejemplo: {"Jacobi": {...}, "SOR": {...}}
        guardar_ruta      : si se especifica, guarda la gráfica en esa ruta

    Retorna:
        figura de matplotlib (para mostrarla en la interfaz con canvas)
    """
    figura, eje = plt.subplots(figsize=(8, 5))
    figura.patch.set_facecolor('#1E1E2E')
    eje.set_facecolor('#2A2A3E')

    for nombre_metodo, resultado in resultados_metodos.items():
        if resultado and resultado.get("historial_error"):
            color = COLORES_METODOS.get(nombre_metodo, "#FFFFFF")
            iteraciones = range(1, len(resultado["historial_error"]) + 1)
            eje.semilogy(
                iteraciones,
                resultado["historial_error"],
                label=f"{nombre_metodo} ({resultado['iteraciones']} iter.)",
                color=color,
                linewidth=2
            )

    eje.set_xlabel("Iteraciones", color="white", fontsize=11)
    eje.set_ylabel("Error (escala log)", color="white", fontsize=11)
    eje.set_title("Convergencia de los métodos numéricos", color="white", fontsize=13)
    eje.legend(facecolor='#1E1E2E', labelcolor='white')
    eje.tick_params(colors='white')
    eje.grid(True, alpha=0.3, color='white')
    for spine in eje.spines.values():
        spine.set_edgecolor('#555555')

    figura.tight_layout()

    if guardar_ruta:
        figura.savefig(guardar_ruta, dpi=150, bbox_inches='tight',
                       facecolor=figura.get_facecolor())
        print(f"[Servicios] Gráfica guardada en: {guardar_ruta}")

    return figura


def generar_mapa_calor(mapa_temperaturas, titulo="Distribución de temperatura",
                       guardar_ruta=None):
    """
    Genera un mapa de calor 2D con colores que van de azul (frío) a rojo (caliente).

    ¿POR QUÉ ESTE COLORMAP?
        'hot' es el colormap estándar para temperatura: negro→rojo→amarillo→blanco.
        Intuitivo para cualquier persona — los colores cálidos = zonas calientes.

    Parámetros:
        mapa_temperaturas: numpy array 2D con temperaturas en °C
        titulo           : título que aparece en la gráfica
        guardar_ruta     : ruta para guardar la imagen PNG (opcional)

    Retorna:
        figura de matplotlib lista para mostrar en interfaz
    """
    figura, eje = plt.subplots(figsize=(6, 5))
    figura.patch.set_facecolor('#1E1E2E')
    eje.set_facecolor('#1E1E2E')

    imagen = eje.imshow(
        mapa_temperaturas,
        cmap='hot',           # colormap: negro→rojo→amarillo→blanco
        interpolation='bilinear',  # suaviza el mapa para que no se vean pixeles cuadrados
        aspect='equal'
    )

    # barra de color con etiqueta de °C
    barra = figura.colorbar(imagen, ax=eje)
    barra.set_label("Temperatura (°C)", color="white")
    barra.ax.yaxis.set_tick_params(color='white')
    plt.setp(barra.ax.yaxis.get_ticklabels(), color='white')

    eje.set_title(titulo, color="white", fontsize=12)
    eje.set_xlabel("Columna (nodo)", color="white")
    eje.set_ylabel("Fila (nodo)", color="white")
    eje.tick_params(colors='white')

    figura.tight_layout()

    if guardar_ruta:
        figura.savefig(guardar_ruta, dpi=150, bbox_inches='tight',
                       facecolor=figura.get_facecolor())
        print(f"[Servicios] Mapa de calor guardado en: {guardar_ruta}")

    return figura


def comparar_mapas_calor(mapas_metodos, guardar_ruta=None):
    """
    Muestra los mapas de calor de los 4 métodos en una sola figura lado a lado.

    ¿PARA QUÉ SIRVE?
        Permite comparar visualmente si los 4 métodos producen resultados
        similares (deberían, si todos convergieron correctamente) y si hay
        diferencias en la distribución de temperatura.

    Parámetros:
        mapas_metodos: dict con nombre_metodo → mapa 2D de temperaturas
        guardar_ruta : ruta para guardar la imagen comparativa (opcional)

    Retorna:
        figura de matplotlib
    """
    n_metodos = len(mapas_metodos)
    figura, ejes = plt.subplots(1, n_metodos, figsize=(5 * n_metodos, 4))
    figura.patch.set_facecolor('#1E1E2E')

    if n_metodos == 1:
        ejes = [ejes]

    for eje, (nombre, mapa) in zip(ejes, mapas_metodos.items()):
        eje.set_facecolor('#1E1E2E')
        im = eje.imshow(mapa, cmap='hot', interpolation='bilinear', aspect='equal')
        eje.set_title(nombre, color="white", fontsize=11)
        eje.tick_params(colors='white')
        figura.colorbar(im, ax=eje).ax.yaxis.set_tick_params(color='white')

    figura.suptitle("Comparación de mapas de calor — 4 métodos",
                    color="white", fontsize=13)
    figura.tight_layout()

    if guardar_ruta:
        figura.savefig(guardar_ruta, dpi=150, bbox_inches='tight',
                       facecolor=figura.get_facecolor())

    return figura


def exportar_csv(resultados_metodos, ruta_archivo):
    """
    Exporta los resultados numéricos de todos los métodos a un archivo CSV.

    ¿QUÉ CONTIENE EL CSV?
        Una fila por método con: nombre, iteraciones, error final, tiempo,
        si convergió y el omega usado (para SOR y SSOR).

    Parámetros:
        resultados_metodos: dict nombre_metodo → resultado del método
        ruta_archivo      : ruta completa del archivo CSV a crear
    """
    encabezados = [
        "Método", "Iteraciones", "Error Final",
        "Tiempo (s)", "Convergió", "Omega"
    ]

    with open(ruta_archivo, mode='w', newline='', encoding='utf-8') as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(encabezados)

        for nombre, resultado in resultados_metodos.items():
            if resultado:
                escritor.writerow([
                    nombre,
                    resultado.get("iteraciones", "-"),
                    f"{resultado.get('error_final', 0):.2e}",
                    f"{resultado.get('tiempo', 0):.4f}",
                    "Sí" if resultado.get("convergio") else "No",
                    resultado.get("omega_usado", "-")
                ])

    print(f"[Servicios] CSV exportado en: {ruta_archivo}")


def exportar_txt(resultados_metodos, ruta_archivo):
    """
    Exporta un informe de texto legible con los resultados de cada método.

    ¿PARA QUÉ EL TXT ADEMÁS DEL CSV?
        El CSV es para procesar datos, el TXT es para leer fácilmente
        o incluir en el informe del proyecto.

    Parámetros:
        resultados_metodos: dict nombre_metodo → resultado del método
        ruta_archivo      : ruta completa del archivo TXT a crear
    """
    marca_tiempo = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(ruta_archivo, mode='w', encoding='utf-8') as archivo:
        archivo.write("=" * 60 + "\n")
        archivo.write("  RESULTADOS — SIMULACIÓN TÉRMICA PANEL SOLAR\n")
        archivo.write(f"  Generado: {marca_tiempo}\n")
        archivo.write("=" * 60 + "\n\n")

        for nombre, resultado in resultados_metodos.items():
            if resultado:
                archivo.write(f"MÉTODO: {nombre}\n")
                archivo.write("-" * 40 + "\n")
                archivo.write(f"  Iteraciones realizadas : {resultado.get('iteraciones', '-')}\n")
                archivo.write(f"  Error final            : {resultado.get('error_final', 0):.2e}\n")
                archivo.write(f"  Tiempo de cómputo      : {resultado.get('tiempo', 0):.4f} s\n")
                archivo.write(f"  Convergió              : {'Sí' if resultado.get('convergio') else 'No'}\n")
                if resultado.get("omega_usado"):
                    archivo.write(f"  Omega (ω) usado        : {resultado.get('omega_usado')}\n")
                archivo.write("\n")

    print(f"[Servicios] TXT exportado en: {ruta_archivo}")


def crear_carpeta_resultados(base="resultados"):
    """
    Crea (si no existe) la carpeta donde se guardan los resultados.
    Le añade una marca de tiempo para no sobreescribir ejecuciones anteriores.

    Retorna:
        ruta de la carpeta creada
    """
    marca = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = os.path.join(base, f"ejecucion_{marca}")
    os.makedirs(ruta, exist_ok=True)
    return ruta