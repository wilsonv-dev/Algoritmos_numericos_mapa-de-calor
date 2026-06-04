# ☀️ Simulación Térmica de Panel Solar
**Proyecto de Aula — Algoritmos Numéricos para Ingeniería | Corte 3**  
Universidad de Pamplona · Docente: Sebastián Echavez Cadena

---

## ¿De qué trata?

Este software simula cómo se distribuye la temperatura en un panel solar fotovoltaico a partir de una fotografía real del panel.

El usuario sube una imagen del panel. El programa la convierte en una malla de nodos de temperatura y resuelve la distribución de calor usando cuatro métodos numéricos iterativos. El resultado es un **mapa de calor** que muestra las zonas frías y calientes del panel, junto con gráficas de convergencia y tablas comparativas.

---

## Modelo matemático

La temperatura en cada nodo satisface la **ecuación de Laplace discreta**:

```
T(i-1,j) + T(i+1,j) + T(i,j-1) + T(i,j+1) - 4·T(i,j) = -Q(i,j)
```

Esto genera un sistema lineal **Ax = b** donde:
- **A** → matriz de coeficientes (diagonal dominante ✅)
- **x** → vector de temperaturas desconocidas
- **b** → condiciones de borde y fuentes de calor

La diagonal dominante garantiza que los 4 métodos **convergen siempre**.

---

## Métodos implementados

| Método | Descripción | Velocidad |
|--------|-------------|-----------|
| Jacobi | Usa solo valores de la iteración anterior | Más lento |
| Gauss-Seidel | Usa valores ya actualizados en la misma iteración | 2x más rápido que Jacobi |
| SOR | Gauss-Seidel con factor de relajación ω | Más rápido con buen ω |
| SSOR | SOR simétrico (barrido adelante + atrás) | Más estable |

---

## Arquitectura — 5 capas

```
proyecto_numerico/
├── main.py                  # Punto de entrada
├── interfaz.py              # Capa 1 — Ventana, botones, gráficas
├── controlador.py           # Capa 2 — Coordinación y validación
├── traduccion.py            # Capa 3 — Imagen → sistema Ax = b
├── servicios.py             # Capa 5 — Exportar CSV, PNG, gráficas
├── metodos_numericos/       # Capa 4 — Métodos numéricos puros
│   ├── jacobi.py
│   ├── gauss_seidel.py
│   ├── sor.py
│   └── ssor.py
├── resultados/              # Se crea automáticamente
└── README.md
```

---

## Requisitos e instalación

```bash
pip install numpy matplotlib pillow
python main.py
```

**Python 3.8 o superior requerido.**

---

## Parámetros del usuario

| Parámetro | Descripción | Ejemplo |
|-----------|-------------|---------|
| Imagen | Foto del panel (JPG, PNG, BMP) | panel.jpg |
| Tolerancia | Criterio de parada | 1e-6 |
| Máx. iteraciones | Límite de iteraciones | 500 |
| Omega ω | Factor de relajación SOR/SSOR, debe estar en (0, 2) | 1.25 |
| Tamaño malla | Resolución NxN de la simulación | 20 |

---

## Resultados que genera

- 🌡️ Mapa de calor con distribución de temperatura
- 📈 Gráfica de Error vs Iteraciones por método
- 📊 Tabla con iteraciones, error final y tiempo de cómputo
- 💾 Exportación a CSV, TXT y PNG

---

## Avance actual

- [x] Capa 4 — 4 métodos implementados y probados
- [x] Capa 3 — Lectura de imagen y construcción de Ax = b
- [x] Capa 2 — Controlador con validación de parámetros
- [x] Capa 5 — Servicios de exportación y graficación
- [x] Capa 1 — Interfaz gráfica con tkinter
- [x] Prueba de integración exitosa
- [ ] Corrección error de importación en Windows
- [ ] Sliders de parámetros opcionales
- [ ] Documentación técnica y manual de usuario

---

## Problema conocido (Windows)

Si aparece este error al ejecutar:
```
ImportError: cannot import name 'gauss_seidel' from 'metodos'
```
**Solución:** renombrar la carpeta `metodos` a `metodos_numericos` y en `controlador.py` cambiar:
```python
# Antes
from metodos import jacobi, gauss_seidel, sor, ssor
# Después
from metodos_numericos import jacobi, gauss_seidel, sor, ssor
```