"""
CAPA 1 - INTERFAZ GRÁFICA
==========================
Esta capa es la única que el usuario ve y toca. Contiene todos los
elementos visuales: botones, sliders, campos de texto y las gráficas.

¿QUÉ NO DEBE HACER ESTA CAPA?
    - No calcula nada matemático
    - No llama directamente a los métodos numéricos
    - No lee imágenes por sí misma
    Solo muestra, recibe input y llama al Controlador (Capa 2).

LIBRERÍA USADA: tkinter (incluida en Python estándar)
    - tk.Tk()         → ventana principal
    - ttk.Frame       → contenedores de secciones
    - ttk.Button      → botones
    - ttk.Entry       → campos de texto
    - ttk.Checkbutton → casillas de selección de métodos
    - FigureCanvasTkAgg → incrusta figuras de matplotlib en tkinter

LAYOUT DE LA INTERFAZ:
    ┌─────────────────────────────────────────────┐
    │  Panel izquierdo     │  Panel derecho        │
    │  - Cargar imagen     │  - Imagen original    │
    │  - Parámetros        │  - Mapa de calor      │
    │  - Selección métodos │  - Gráfica convergencia│
    │  - Botones acción    │  - Tabla de resultados │
    └─────────────────────────────────────────────┘
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import sys

import matplotlib
matplotlib.use("TkAgg")   # backend de matplotlib para tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk

# importamos las capas que necesitamos
import controlador
import servicios


# ─────────────────────────────────────────────
# COLORES DE LA INTERFAZ (tema oscuro)
# ─────────────────────────────────────────────
COLOR_FONDO        = "#1E1E2E"
COLOR_PANEL        = "#2A2A3E"
COLOR_ACENTO       = "#7C3AED"    # morado
COLOR_ACENTO2      = "#10B981"    # verde
COLOR_TEXTO        = "#E2E8F0"
COLOR_TEXTO_CLARO  = "#94A3B8"
COLOR_BOTON        = "#3730A3"
COLOR_BOTON_HOVER  = "#4338CA"
COLOR_ERROR        = "#EF4444"
COLOR_EXITO        = "#10B981"


class AplicacionPanelSolar:
    """
    Clase principal de la interfaz. Encapsula toda la lógica visual.

    ¿POR QUÉ UNA CLASE?
        Mantener el estado de la aplicación (ruta de imagen, resultados,
        widgets) en atributos de instancia evita variables globales y
        hace el código más organizado y mantenible.
    """

    def __init__(self, ventana_raiz):
        """
        Constructor: configura la ventana principal y construye todos
        los widgets de la interfaz.
        """
        self.ventana = ventana_raiz
        self.ventana.title("Simulación Térmica — Panel Solar ☀️")
        self.ventana.configure(bg=COLOR_FONDO)
        self.ventana.geometry("1280x800")
        self.ventana.minsize(1000, 650)

        # ── Estado interno de la aplicación ──
        self.ruta_imagen      = None     # ruta al archivo de imagen cargado
        self.resultados_sim   = None     # resultados de la última simulación
        self.canvas_original  = None     # canvas del mapa original
        self.canvas_calor     = None     # canvas del mapa de calor resultante
        self.canvas_conv      = None     # canvas de la gráfica de convergencia

        # ── Construir la interfaz ──
        self._configurar_estilos()
        self._construir_layout()

    # ─────────────────────────────────────────────
    # CONSTRUCCIÓN DEL LAYOUT
    # ─────────────────────────────────────────────

    def _configurar_estilos(self):
        """Configura los estilos visuales de los widgets ttk."""
        estilo = ttk.Style()
        estilo.theme_use("clam")

        estilo.configure("TFrame",        background=COLOR_FONDO)
        estilo.configure("Panel.TFrame",  background=COLOR_PANEL)
        estilo.configure("TLabel",        background=COLOR_FONDO,
                         foreground=COLOR_TEXTO, font=("Segoe UI", 10))
        estilo.configure("Titulo.TLabel", background=COLOR_FONDO,
                         foreground=COLOR_TEXTO, font=("Segoe UI", 12, "bold"))
        estilo.configure("TCheckbutton",  background=COLOR_PANEL,
                         foreground=COLOR_TEXTO, font=("Segoe UI", 10))
        estilo.configure("TEntry",        fieldbackground="#3A3A5C",
                         foreground=COLOR_TEXTO, insertcolor=COLOR_TEXTO)
        estilo.configure("Accion.TButton", background=COLOR_BOTON,
                         foreground="white", font=("Segoe UI", 10, "bold"),
                         padding=8)
        estilo.map("Accion.TButton",
                   background=[("active", COLOR_BOTON_HOVER)])

    def _construir_layout(self):
        """Construye el layout principal dividido en panel izquierdo y derecho."""

        # ── Marco principal ──
        marco_principal = ttk.Frame(self.ventana)
        marco_principal.pack(fill="both", expand=True, padx=10, pady=10)
        marco_principal.columnconfigure(0, weight=1, minsize=300)
        marco_principal.columnconfigure(1, weight=3)
        marco_principal.rowconfigure(0, weight=1)

        # ── Panel izquierdo: controles ──
        self.panel_izq = ttk.Frame(marco_principal, style="Panel.TFrame")
        self.panel_izq.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        # ── Panel derecho: visualizaciones ──
        self.panel_der = ttk.Frame(marco_principal)
        self.panel_der.grid(row=0, column=1, sticky="nsew")
        self.panel_der.columnconfigure(0, weight=1)
        self.panel_der.columnconfigure(1, weight=1)
        self.panel_der.rowconfigure(0, weight=1)
        self.panel_der.rowconfigure(1, weight=1)

        self._construir_panel_controles()
        self._construir_panel_visualizaciones()

    def _construir_panel_controles(self):
        """Construye el panel izquierdo con todos los controles del usuario."""
        panel = self.panel_izq
        pad = {"padx": 12, "pady": 6}

        # ── Título ──
        ttk.Label(panel, text="☀️  Panel Solar — Simulación Térmica",
                  style="Titulo.TLabel",
                  background=COLOR_PANEL).pack(fill="x", **pad, pady=(14, 4))

        ttk.Separator(panel, orient="horizontal").pack(fill="x", padx=12, pady=4)

        # ── Sección: Cargar imagen ──
        ttk.Label(panel, text="1. IMAGEN DEL PANEL",
                  background=COLOR_PANEL,
                  foreground=COLOR_ACENTO,
                  font=("Segoe UI", 9, "bold")).pack(anchor="w", **pad)

        self.lbl_imagen = ttk.Label(panel, text="Ninguna imagen cargada",
                                    background=COLOR_PANEL,
                                    foreground=COLOR_TEXTO_CLARO,
                                    font=("Segoe UI", 9, "italic"),
                                    wraplength=240)
        self.lbl_imagen.pack(anchor="w", **pad)

        ttk.Button(panel, text="📂  Cargar Imagen",
                   style="Accion.TButton",
                   command=self._cargar_imagen).pack(fill="x", **pad)

        ttk.Separator(panel, orient="horizontal").pack(fill="x", padx=12, pady=6)

        # ── Sección: Parámetros ──
        ttk.Label(panel, text="2. PARÁMETROS",
                  background=COLOR_PANEL,
                  foreground=COLOR_ACENTO,
                  font=("Segoe UI", 9, "bold")).pack(anchor="w", **pad)

        self._crear_campo(panel, "Tolerancia:", "tolerancia_var", "1e-6")
        self._crear_campo(panel, "Máx. iteraciones:", "iteraciones_var", "500")
        self._crear_campo(panel, "Omega ω (SOR/SSOR):", "omega_var", "1.25")
        self._crear_campo(panel, "Tamaño malla (NxN):", "malla_var", "20")

        ttk.Separator(panel, orient="horizontal").pack(fill="x", padx=12, pady=6)

        # ── Sección: Selección de métodos ──
        ttk.Label(panel, text="3. MÉTODOS A EJECUTAR",
                  background=COLOR_PANEL,
                  foreground=COLOR_ACENTO,
                  font=("Segoe UI", 9, "bold")).pack(anchor="w", **pad)

        self.check_jacobi    = self._crear_check(panel, "Jacobi",       True)
        self.check_gauss     = self._crear_check(panel, "Gauss-Seidel", True)
        self.check_sor       = self._crear_check(panel, "SOR",          True)
        self.check_ssor      = self._crear_check(panel, "SSOR",         True)

        ttk.Separator(panel, orient="horizontal").pack(fill="x", padx=12, pady=6)

        # ── Botones de acción ──
        ttk.Label(panel, text="4. ACCIONES",
                  background=COLOR_PANEL,
                  foreground=COLOR_ACENTO,
                  font=("Segoe UI", 9, "bold")).pack(anchor="w", **pad)

        ttk.Button(panel, text="▶  Calcular",
                   style="Accion.TButton",
                   command=self._ejecutar_simulacion).pack(fill="x", **pad)

        ttk.Button(panel, text="💾  Guardar Resultados",
                   style="Accion.TButton",
                   command=self._guardar_resultados).pack(fill="x", **pad)

        ttk.Button(panel, text="🗑  Limpiar",
                   style="Accion.TButton",
                   command=self._limpiar).pack(fill="x", **pad)

        ttk.Button(panel, text="✖  Salir",
                   style="Accion.TButton",
                   command=self.ventana.quit).pack(fill="x", **pad)

        # ── Barra de estado ──
        self.lbl_estado = ttk.Label(panel, text="Listo.",
                                    background=COLOR_PANEL,
                                    foreground=COLOR_TEXTO_CLARO,
                                    font=("Segoe UI", 9, "italic"),
                                    wraplength=260)
        self.lbl_estado.pack(anchor="w", padx=12, pady=(10, 6))

    def _crear_campo(self, padre, etiqueta, nombre_var, valor_default):
        """
        Crea un campo de entrada con su etiqueta.
        Guarda la variable tkinter como atributo de instancia con el nombre dado.
        """
        marco = ttk.Frame(padre, style="Panel.TFrame")
        marco.pack(fill="x", padx=12, pady=2)

        ttk.Label(marco, text=etiqueta,
                  background=COLOR_PANEL,
                  foreground=COLOR_TEXTO,
                  font=("Segoe UI", 9)).pack(anchor="w")

        var = tk.StringVar(value=valor_default)
        setattr(self, nombre_var, var)   # self.tolerancia_var = var

        entrada = ttk.Entry(marco, textvariable=var, font=("Segoe UI", 10))
        entrada.pack(fill="x")

    def _crear_check(self, padre, texto, valor_default):
        """Crea una casilla de verificación para seleccionar un método."""
        var = tk.BooleanVar(value=valor_default)
        ttk.Checkbutton(padre, text=texto, variable=var,
                        style="TCheckbutton").pack(anchor="w", padx=16, pady=1)
        return var

    def _construir_panel_visualizaciones(self):
        """Construye el panel derecho con los 4 cuadrantes de visualización."""
        panel = self.panel_der

        # cuadrante superior izquierdo: imagen original
        self.marco_original = self._crear_cuadrante(
            panel, "Imagen Original (escala de grises)", 0, 0
        )

        # cuadrante superior derecho: mapa de calor resultado
        self.marco_calor = self._crear_cuadrante(
            panel, "Mapa de Calor — Temperatura (°C)", 0, 1
        )

        # cuadrante inferior izquierdo: gráfica de convergencia
        self.marco_convergencia = self._crear_cuadrante(
            panel, "Convergencia — Error vs Iteraciones", 1, 0
        )

        # cuadrante inferior derecho: tabla de resultados numéricos
        self.marco_tabla = self._crear_cuadrante(
            panel, "Resultados Numéricos", 1, 1
        )
        self._construir_tabla_resultados(self.marco_tabla)

    def _crear_cuadrante(self, padre, titulo, fila, columna):
        """Crea un cuadrante con título en el panel de visualizaciones."""
        marco = ttk.Frame(padre, style="Panel.TFrame")
        marco.grid(row=fila, column=columna, sticky="nsew", padx=4, pady=4)
        marco.columnconfigure(0, weight=1)
        marco.rowconfigure(1, weight=1)

        ttk.Label(marco, text=titulo,
                  background=COLOR_PANEL,
                  foreground=COLOR_ACENTO,
                  font=("Segoe UI", 9, "bold")).grid(
                      row=0, column=0, sticky="w", padx=8, pady=4
                  )
        return marco

    def _construir_tabla_resultados(self, padre):
        """Construye la tabla Treeview para mostrar resultados numéricos."""
        columnas = ("Método", "Iteraciones", "Error Final", "Tiempo (s)", "Convergió")

        self.tabla = ttk.Treeview(padre, columns=columnas, show="headings", height=6)

        for col in columnas:
            self.tabla.heading(col, text=col)
            self.tabla.column(col, width=90, anchor="center")

        # estilo de la tabla
        estilo = ttk.Style()
        estilo.configure("Treeview",
                         background=COLOR_PANEL,
                         foreground=COLOR_TEXTO,
                         fieldbackground=COLOR_PANEL,
                         font=("Segoe UI", 9))
        estilo.configure("Treeview.Heading",
                         background=COLOR_BOTON,
                         foreground="white",
                         font=("Segoe UI", 9, "bold"))

        self.tabla.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)

    # ─────────────────────────────────────────────
    # ACCIONES DE LOS BOTONES
    # ─────────────────────────────────────────────

    def _cargar_imagen(self):
        """Abre el diálogo para seleccionar una imagen y la muestra en la interfaz."""
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen del panel solar",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("Todos los archivos", "*.*")
            ]
        )

        if not ruta:
            return   # el usuario canceló

        self.ruta_imagen = ruta
        nombre_archivo = os.path.basename(ruta)
        self.lbl_imagen.config(text=f"✓ {nombre_archivo}")
        self._actualizar_estado(f"Imagen cargada: {nombre_archivo}")

        # mostramos una vista previa de la imagen original en escala de grises
        self._mostrar_preview_imagen(ruta)

    def _mostrar_preview_imagen(self, ruta):
        """Muestra la imagen cargada en escala de grises en el cuadrante superior izquierdo."""
        try:
            tamanio = int(self.malla_var.get())
        except ValueError:
            tamanio = 20

        # abrimos y convertimos a escala de grises
        from PIL import Image
        img = Image.open(ruta).convert('L').resize((tamanio * 10, tamanio * 10), Image.LANCZOS)
        img_array = np.array(img)

        # creamos la figura matplotlib
        fig = Figure(figsize=(3, 3), facecolor=COLOR_PANEL)
        ax = fig.add_subplot(111)
        ax.imshow(img_array, cmap='gray', aspect='equal')
        ax.set_title("Imagen original (grises)", color=COLOR_TEXTO, fontsize=9)
        ax.axis('off')
        fig.tight_layout()

        self._mostrar_figura_en_cuadrante(fig, self.marco_original, "canvas_original")

    def _ejecutar_simulacion(self):
        """
        Ejecuta la simulación en un hilo separado para no congelar la interfaz.

        ¿POR QUÉ UN HILO SEPARADO (threading)?
            Los métodos iterativos pueden tardar varios segundos.
            Si los ejecutamos en el hilo principal, la ventana se congela
            y el usuario piensa que el programa falló.
        """
        if not self.ruta_imagen:
            messagebox.showwarning("Advertencia", "Primero debes cargar una imagen.")
            return

        # recopilamos qué métodos seleccionó el usuario
        metodos = []
        if self.check_jacobi.get():    metodos.append("Jacobi")
        if self.check_gauss.get():     metodos.append("Gauss-Seidel")
        if self.check_sor.get():       metodos.append("SOR")
        if self.check_ssor.get():      metodos.append("SSOR")

        if not metodos:
            messagebox.showwarning("Advertencia", "Selecciona al menos un método.")
            return

        # leemos y convertimos los parámetros
        try:
            tolerancia     = float(self.tolerancia_var.get())
            max_iter       = int(self.iteraciones_var.get())
            omega          = float(self.omega_var.get())
            tamanio_malla  = int(self.malla_var.get())
        except ValueError:
            messagebox.showerror("Error", "Los parámetros deben ser valores numéricos válidos.")
            return

        self._actualizar_estado("⏳ Ejecutando simulación...")

        # ejecutamos en hilo secundario
        hilo = threading.Thread(
            target=self._hilo_simulacion,
            args=(metodos, tolerancia, max_iter, omega, tamanio_malla),
            daemon=True
        )
        hilo.start()

    def _hilo_simulacion(self, metodos, tolerancia, max_iter, omega, tamanio_malla):
        """
        Función que corre en el hilo secundario.
        Llama al Controlador y después actualiza la interfaz.
        """
        resultado = controlador.ejecutar_simulacion(
            ruta_imagen=self.ruta_imagen,
            metodos_seleccionados=metodos,
            tolerancia=tolerancia,
            max_iteraciones=max_iter,
            omega=omega,
            tamanio_malla=tamanio_malla
        )

        # actualizamos la interfaz desde el hilo principal (tkinter no es thread-safe)
        self.ventana.after(0, self._mostrar_resultados, resultado)

    def _mostrar_resultados(self, resultado):
        """Actualiza todos los cuadrantes visuales con los resultados de la simulación."""

        if resultado.get("error"):
            messagebox.showerror("Error en la simulación", resultado["error"])
            self._actualizar_estado("❌ Error en la simulación.")
            return

        self.resultados_sim = resultado

        mapas     = resultado["mapas_calor"]
        resultados = resultado["resultados"]

        # ── Mapa de calor: mostramos el primer método disponible ──
        if mapas:
            primer_metodo = list(mapas.keys())[0]
            fig_calor = servicios.generar_mapa_calor(
                mapas[primer_metodo],
                titulo=f"Temperatura — {primer_metodo}"
            )
            self._mostrar_figura_en_cuadrante(fig_calor, self.marco_calor, "canvas_calor")

        # ── Gráfica de convergencia ──
        fig_conv = servicios.graficar_convergencia(resultados)
        self._mostrar_figura_en_cuadrante(fig_conv, self.marco_convergencia, "canvas_conv")

        # ── Tabla de resultados numéricos ──
        self._actualizar_tabla(resultados)

        self._actualizar_estado("✅ Simulación completada.")

    def _actualizar_tabla(self, resultados):
        """Actualiza la tabla de resultados numéricos con los datos de cada método."""
        # limpiamos filas anteriores
        for fila in self.tabla.get_children():
            self.tabla.delete(fila)

        for nombre, res in resultados.items():
            if res and not res.get("error"):
                self.tabla.insert("", "end", values=(
                    nombre,
                    res.get("iteraciones", "-"),
                    f"{res.get('error_final', 0):.2e}",
                    f"{res.get('tiempo', 0):.3f}",
                    "✓ Sí" if res.get("convergio") else "✗ No"
                ))

    def _mostrar_figura_en_cuadrante(self, figura, cuadrante, nombre_canvas):
        """
        Incrusta una figura de matplotlib dentro de un cuadrante tkinter.

        ¿CÓMO FUNCIONA FigureCanvasTkAgg?
            Es el puente entre matplotlib y tkinter. Convierte la figura
            matplotlib en un widget que puede vivir dentro de una ventana tkinter.
        """
        # destruimos el canvas anterior si existe
        canvas_anterior = getattr(self, nombre_canvas, None)
        if canvas_anterior:
            canvas_anterior.get_tk_widget().destroy()

        canvas = FigureCanvasTkAgg(figura, master=cuadrante)
        canvas.draw()
        canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        setattr(self, nombre_canvas, canvas)
        plt.close(figura)  # liberamos memoria de la figura

    def _guardar_resultados(self):
        """Guarda el CSV, TXT y el mapa de calor en una carpeta de resultados."""
        if not self.resultados_sim:
            messagebox.showwarning("Advertencia", "Primero ejecuta una simulación.")
            return

        carpeta = servicios.crear_carpeta_resultados()

        # exportar CSV y TXT
        servicios.exportar_csv(
            self.resultados_sim["resultados"],
            os.path.join(carpeta, "resultados.csv")
        )
        servicios.exportar_txt(
            self.resultados_sim["resultados"],
            os.path.join(carpeta, "resultados.txt")
        )

        # guardar mapas de calor
        for nombre, mapa in self.resultados_sim["mapas_calor"].items():
            nombre_archivo = nombre.replace("-", "_").replace(" ", "_").lower()
            servicios.generar_mapa_calor(
                mapa,
                titulo=f"Temperatura — {nombre}",
                guardar_ruta=os.path.join(carpeta, f"mapa_{nombre_archivo}.png")
            )
            plt.close('all')

        # guardar gráfica de convergencia
        servicios.graficar_convergencia(
            self.resultados_sim["resultados"],
            guardar_ruta=os.path.join(carpeta, "convergencia.png")
        )
        plt.close('all')

        messagebox.showinfo("Guardado", f"Resultados guardados en:\n{carpeta}")
        self._actualizar_estado(f"💾 Guardado en: {carpeta}")

    def _limpiar(self):
        """Limpia todos los resultados y reinicia la interfaz."""
        self.ruta_imagen    = None
        self.resultados_sim = None
        self.lbl_imagen.config(text="Ninguna imagen cargada")
        self._actualizar_estado("Listo.")

        # limpiamos los canvases de visualización
        for nombre_canvas in ["canvas_original", "canvas_calor", "canvas_conv"]:
            c = getattr(self, nombre_canvas, None)
            if c:
                c.get_tk_widget().destroy()
                setattr(self, nombre_canvas, None)

        # limpiamos la tabla
        for fila in self.tabla.get_children():
            self.tabla.delete(fila)

    def _actualizar_estado(self, mensaje):
        """Actualiza el label de estado en el panel izquierdo."""
        self.lbl_estado.config(text=mensaje)
        self.ventana.update_idletasks()


# ─────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────

def iniciar_aplicacion():
    """Crea la ventana principal e inicia el loop de tkinter."""
    ventana = tk.Tk()
    app = AplicacionPanelSolar(ventana)
    ventana.mainloop()