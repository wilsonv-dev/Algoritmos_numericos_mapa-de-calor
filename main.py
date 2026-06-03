"""
MAIN.PY — PUNTO DE ENTRADA
===========================
Este archivo es el único que el usuario ejecuta directamente:
    python main.py

¿POR QUÉ UN main.py SEPARADO?
    Mantener el punto de entrada separado de la interfaz permite
    en el futuro ejecutar el proyecto sin interfaz (modo consola),
    hacer pruebas automatizadas, o cambiar la interfaz sin tocar
    la lógica de arranque.
"""

import sys
import os

# aseguramos que Python encuentre los módulos del proyecto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interfaz import iniciar_aplicacion


if __name__ == "__main__":
    iniciar_aplicacion()