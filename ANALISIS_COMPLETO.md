# ANALISIS COMPLETO DEL REPOSITORIO
## Neural Canvas v2.1 - Computer Vision Interactive Art Application

**Fecha de Analisis:** 2025-12-14
**Analizado por:** Claude (Senior Engineer)
**Contexto:** Ubuntu/Linux | numpy==1.26.4, opencv-python==4.8.1.78, mediapipe==0.10.21

---

## 1. MAPA DEL REPOSITORIO

### 1.1 Estructura de Archivos

```
/proyectosFinalesGraficacion/
├── main.py              # ENTRYPOINT - Controlador principal (447 lineas)
├── Gesture3D.py         # Modulo de reconocimiento de gestos (651 lineas)
├── ColorPainter.py      # Modulo de color tracking y pintura (253 lineas)
├── neon_effects.py      # Efectos visuales neon (165 lineas) [NO USADO]
├── requirements.txt     # Dependencias (3 paquetes)
└── .gitignore           # Exclusiones de git
```

**Total:** 1,516 lineas de Python

---

### 1.2 Descripcion de Cada Archivo

#### `main.py` - ENTRYPOINT PRINCIPAL

| Clase/Funcion | Linea | Descripcion |
|---------------|-------|-------------|
| `class PizarraNeon` | 13 | Clase principal que orquesta toda la aplicacion |
| `__init__()` | 14-47 | Inicializa camara, modulos, cache, FPS counter, paleta de colores |
| `inicializar()` | 49-70 | Abre VideoCapture(0), configura 1280x720@30fps, buffer=1 |
| `dibujar_texto_limpio()` | 72-79 | Texto con sombra sutil (estatico) |
| `dibujar_grid_minimal()` | 81-105 | Grid animado con cache (0.2s interval) |
| `dibujar_borde_esquinas()` | 107-140 | HUD esquinas con pulso animado |
| `dibujar_display_seleccion()` | 142-171 | Header verde cuando figura seleccionada |
| `dibujar_menu_principal()` | 212-266 | Menu modo "menu" con opciones 1,2,Q |
| `dibujar_hud_superior()` | 268-278 | Barra superior con modo y FPS |
| `modo_seguimiento_color()` | 280-287 | Delega a ColorPainter.process_frame() |
| `modo_figuras_gestos()` | 289-299 | Delega a Gesture3D.process_frame() |
| `procesar_teclas()` | 301-325 | Switch principal de teclas |
| `_procesar_teclas_color()` | 327-340 | Teclas modo color: SPACE, C, +, - |
| `_procesar_teclas_gestos()` | 342-368 | Teclas modo gestos: 1-5, X, SPACE, S |
| `ejecutar()` | 370-433 | **BUCLE PRINCIPAL** - captura, procesa, muestra |
| `liberar_recursos()` | 435-442 | Limpieza al cerrar |

**Flujo de ejecucion:**
```
__main__ → PizarraNeon() → ejecutar() → inicializar() → while True:
    ├── cap.read()
    ├── cv2.flip(frame, 1)  # Espejo horizontal
    ├── if modo == "menu": dibujar_menu_principal()
    ├── elif modo == "color": modo_seguimiento_color() → ColorPainter
    ├── elif modo == "gestos": modo_figuras_gestos() → Gesture3D
    ├── cv2.imshow()
    └── procesar_teclas()
```

---

#### `Gesture3D.py` - RECONOCIMIENTO DE GESTOS

| Clase/Enum | Linea | Descripcion |
|------------|-------|-------------|
| `Gesture(Enum)` | 8-13 | NONE=0, FIST=1, OPEN_HAND=2, PINCH=3, VICTORY=4 |
| `MenuState(Enum)` | 16-18 | HIDDEN=0, VISIBLE=1 |
| `SelectionMode(Enum)` | 21-23 | NORMAL=0, SCALE=1 |
| `class Gesture3D` | 26 | Clase principal de gestos |

**Metodos clave de Gesture3D:**

| Metodo | Linea | Descripcion |
|--------|-------|-------------|
| `__init__()` | 27-91 | Estado: figures[], selected_figure, pinch_*, menu_*, colors[], MediaPipe |
| `_initialize_mediapipe()` | 93-109 | Hands(max_num_hands=1, confidence=0.7) |
| `_get_pixel_landmarks()` | 111-114 | Normalizado → px: `int(lm.x * width)` |
| `_get_finger_states()` | 116-138 | Detecta 5 dedos extendidos [pulgar, indice, medio, anular, menique] |
| **`detect_gestures()`** | 140-211 | **DETECCION DE GESTOS** - retorna (gesture, pinch_pos, landmarks) |
| **`handle_gestures()`** | 213-257 | **LOGICA DE GESTOS** - menu, pinch, rotacion |
| `handle_figure_scaling_by_fingers()` | 259-286 | Escala por distancia pulgar-indice |
| `handle_menu_selection()` | 296-338 | Seleccion de item en menu circular |
| `handle_figure_selection()` | 340-366 | Selecciona figura cercana o crea nueva |
| `create_figure()` | 368-383 | Crea dict con type, position, size, color, rotation |
| `move_figure()` | 391-398 | Mueve selected_figure a new_position |
| **`rotate_figure()`** | 400-403 | **ROTACION ACTUAL** - `rotation += 6` |
| `draw_figures()` | 561-564 | Itera y dibuja todas las figuras |
| `_draw_single_figure()` | 566-605 | Dibuja segun type (circle, square, triangle, star, heart, hexagon) |
| `process_frame()` | 634-651 | Orquesta: detect → handle → draw |

---

#### `ColorPainter.py` - TRACKING DE COLOR Y PINTURA

| Metodo | Linea | Descripcion |
|--------|-------|-------------|
| `__init__()` | 7-34 | Canvas vacio, brush_size=15, paleta 8 colores, cache, particulas |
| **`detect_blue_object()`** | 36-74 | **COLOR TRACKING** - HSV, findContours, moments → centroide |
| `add_paint_effect()` | 76-96 | Crea particulas con velocidad/lifetime |
| `update_paint_effects()` | 98-119 | Fisica de particulas (movimiento, friccion) |
| `draw_paint_effects()` | 121-130 | Renderiza particulas con alpha fade |
| `draw_on_canvas()` | 132-151 | cv2.line() entre last_pos y current_pos |
| `clear_canvas()` | 153-157 | Reset canvas a zeros |
| `change_brush_color()` | 159-163 | Cicla paleta de 8 colores |
| `change_brush_size()` | 165-171 | +/- 2px, limites [3, 60] |
| `draw_ui_elements_mejorados()` | 173-227 | Panel inferior con info brush |
| `process_frame()` | 229-253 | detect → draw_on_canvas → addWeighted → UI |

---

#### `neon_effects.py` - EFECTOS VISUALES (NO USADO ACTIVAMENTE)

| Metodo | Linea | Descripcion |
|--------|-------|-------------|
| `draw_glowing_text()` | 13-31 | Texto con glow minimalista |
| `draw_cyber_grid()` | 33-53 | Grid azul pulsante |
| `draw_matrix_rain()` | 55-68 | Lluvia estilo Matrix |
| `draw_pulsating_border()` | 70-111 | Borde esquinas con pulso |
| `create_scan_lines()` | 113-123 | Lineas CRT |
| `draw_terminal_cursor()` | 125-130 | Cursor parpadeante |
| `draw_hex_pattern()` | 132-153 | Patron hexagonal |
| `draw_data_stream()` | 155-166 | Lineas de datos animadas |

**NOTA:** Este modulo esta importado pero NO se usa en el codigo actual. main.py implementa sus propios efectos inline.

---

## 2. IMPLEMENTACION ACTUAL DETALLADA

### 2.1 Color Tracking y Landmark (Centroide)

**Archivo:** `ColorPainter.py:36-74`

```python
def detect_blue_object(self, frame):
    # 1. Conversion a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. Mascara para azul
    lower_blue = np.array([100, 120, 50])   # H=100-130 (azul)
    upper_blue = np.array([130, 255, 255])  # S=120-255, V=50-255
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 3. Morfologia (limpiar ruido)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Erosion+Dilatacion
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Dilatacion+Erosion

    # 4. Encontrar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. Contorno mas grande (area > 300px)
    largest_contour = max(contours, key=cv2.contourArea)

    # 6. CENTROIDE via Momentos
    M = cv2.moments(largest_contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), largest_contour
```

**Problemas actuales:**
- Sin suavizado: el centroide salta frame a frame (jitter visible)
- Cache de mascara cada 0.1s ayuda rendimiento pero NO reduce jitter
- Umbral fijo HSV: sensible a iluminacion

---

### 2.2 Suavizado Actual y Jitter

**Estado actual:** NO HAY SUAVIZADO DEL LANDMARK

En `ColorPainter.py`:
- `last_pos` solo se usa para dibujar lineas entre puntos consecutivos
- NO hay filtrado (EMA, Kalman, etc.) aplicado a `current_pos`

En `Gesture3D.py`:
- `size_smoothing = 0.2` (linea 68) - Solo para escala de figuras
- `_apply_smoothing()` (linea 288-294) - Solo para factor de escala
- Las posiciones de mano (`pinch_position`) NO tienen suavizado

**Jitter observable:**
1. El pincel "tiembla" al pintar lineas
2. Las figuras "saltan" al moverlas con pinch
3. El cursor de tracking vibra constantemente

---

### 2.3 Modos de Operacion

**Archivo:** `main.py` - `self.modo_actual`

| Modo | Valor | Activado por | Descripcion |
|------|-------|--------------|-------------|
| Menu | `"menu"` | Tecla `M` | Pantalla de seleccion |
| Color | `"color"` | Tecla `1` | Color tracking + pintura libre |
| Gestos | `"gestos"` | Tecla `2` | Figuras + gestos de mano |

**Trazo libre (modo color):**
- ColorPainter dibuja lineas continuas en `self.canvas`
- `draw_on_canvas()` usa `cv2.line()` entre `last_pos` y `current_pos`

**Figuras (modo gestos):**
- Lista de figuras en `Gesture3D.figures[]`
- Cada figura es un dict: `{type, position, size, color, rotation, ...}`
- Se dibujan con primitivas OpenCV (circle, rectangle, polylines)

---

### 2.4 Borrado del Lienzo

**Modo Color:**
- Tecla `SPACE` → `ColorPainter.clear_canvas()` (linea 153-157)
```python
def clear_canvas(self):
    self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
    self.last_pos = None
    self.trail_particles.clear()
```

**Modo Gestos:**
- Tecla `X` → `Gesture3D.clear_figures()` (linea 413-418)
- Tecla `SPACE` → `Gesture3D.delete_selected_figure()` (linea 405-411)

---

### 2.5 Menu de Figuras

**Archivo:** `Gesture3D.py`

**Figuras disponibles:** (linea 52-54)
```python
self.available_figures = ['circle', 'square', 'triangle', 'star', 'heart', 'hexagon']
```

**Activacion del menu:** Gesto VICTORY (2 dedos extendidos)
```python
# Linea 216-221
if (gesture == Gesture.VICTORY and
    current_time - self.last_victory_time > self.gesture_cooldown):
    self.menu_state = MenuState.VISIBLE if self.menu_state == MenuState.HIDDEN else MenuState.HIDDEN
```

**Renderizado:** Menu circular (linea 454-524)
- Radio: 200px
- 6 botones de figuras en circulo
- 2 botones centrales: COLOR y DELETE

**PROBLEMA CON ROTACION:**
Las figuras TIENEN propiedad `rotation` pero **NO SE APLICA VISUALMENTE** al dibujar:

```python
# Linea 578-596 - _draw_single_figure()
if figure['type'] == 'circle':
    cv2.circle(frame, pos, size, color, 3, cv2.LINE_AA)  # SIN ROTACION
elif figure['type'] == 'square':
    cv2.rectangle(frame, (pos[0]-size, pos[1]-size), ...)  # SIN ROTACION
elif figure['type'] == 'triangle':
    pts = np.array([...])  # PUNTOS FIJOS, SIN ROTAR
```

La rotacion solo se muestra como una linea indicadora:
```python
# Linea 601-605
if figure['rotation'] != 0:
    angle_rad = math.radians(figure['rotation'])
    end_x = int(pos[0] + size * 0.8 * math.cos(angle_rad))
    end_y = int(pos[1] + size * 0.8 * math.sin(angle_rad))
    cv2.line(frame, pos, (end_x, end_y), color, 2)  # SOLO UNA LINEA
```

---

### 2.6 Gestos Implementados

| Gesto | Deteccion | Accion |
|-------|-----------|--------|
| **PINCH** | `finger_distance < hand_width * 0.3` | Seleccionar/Mover/Escalar figura |
| **VICTORY** | Indice + Medio extendidos (sum=2) | Toggle menu circular |
| **FIST** | 0 dedos extendidos | Ninguna accion |
| **OPEN_HAND** | 4+ dedos extendidos | Incrementar rotacion |

**Deteccion de dedos extendidos:** `_get_finger_states()` (linea 116-138)
```python
# Pulgar: compara tip.x vs ip.x vs mcp.x segun mano izq/der
# Otros dedos: tip.y < pip.y (punta arriba del nudillo)
fingers = [thumb, index, middle, ring, pinky]  # 0 o 1 cada uno
```

---

## 3. DETECCION DE "MANO ABIERTA" Y "ROTACION FAKE"

### 3.1 Deteccion de Mano Abierta

**Ubicacion exacta:** `Gesture3D.py:201-202`

```python
elif extended_count >= 4:
    gesture = Gesture.OPEN_HAND
```

**Flujo completo:**
1. `detect_gestures()` llama a `_get_finger_states()` (linea 168)
2. Cuenta dedos: `extended_count = sum(fingers)` (linea 169)
3. Si 4 o mas dedos extendidos → `OPEN_HAND`

**Condicion:** No debe estar haciendo pinch primero (linea 193-194 tiene prioridad)

---

### 3.2 Rotacion Actual (FAKE)

**Ubicacion exacta:** `Gesture3D.py:253-257`

```python
# Rotacion con mano abierta
if (gesture == Gesture.OPEN_HAND and self.selected_figure and
        current_time - self.last_open_hand_time > 0.15):  # Cooldown 150ms
    self.rotate_figure()
    self.last_open_hand_time = current_time
```

**Funcion rotate_figure():** (linea 400-403)
```python
def rotate_figure(self):
    if self.selected_figure:
        self.selected_figure['rotation'] = (self.selected_figure['rotation'] + 6) % 360
```

**Por que es FAKE:**
1. Solo incrementa un numero (`rotation += 6`)
2. Este angulo NO se aplica a los vertices de la figura
3. Solo dibuja una linea indicadora desde el centro
4. La figura visualmente NO ROTA

**Para rotacion REAL necesitamos:**
```python
# Transformar cada vertice alrededor del centroide:
# x' = cx + (x - cx) * cos(θ) - (y - cy) * sin(θ)
# y' = cy + (x - cx) * sin(θ) + (y - cy) * cos(θ)
```

---

## 4. PLAN DE CAMBIOS - PROYECTO 1 (Detallado)

### 4.1 Estructura Propuesta

```
proyecto1/
├── README.md                 # Documentacion del proyecto
├── requirements.txt          # Dependencias
├── src/
│   ├── __init__.py
│   ├── main.py              # Entrypoint
│   ├── core/
│   │   ├── __init__.py
│   │   ├── app.py           # Clase PizarraNeon refactorizada
│   │   └── config.py        # Configuracion centralizada
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── color_detector.py    # Deteccion de color
│   │   ├── gesture_detector.py  # Deteccion de gestos
│   │   └── filters.py           # EMA / Kalman
│   ├── drawing/
│   │   ├── __init__.py
│   │   ├── canvas.py        # Canvas de pintura
│   │   ├── figures.py       # Figuras geometricas
│   │   └── effects.py       # Efectos visuales
│   └── ui/
│       ├── __init__.py
│       ├── menu.py          # Menu principal
│       └── hud.py           # HUD y overlays
└── tests/
    ├── __init__.py
    ├── conftest.py          # Fixtures pytest
    ├── test_color_detector.py
    ├── test_gesture_detector.py
    ├── test_figures.py
    ├── test_filters.py
    └── mocks/
        ├── __init__.py
        └── frames.py        # Frames simulados para tests
```

---

### 4.2 Checklist de Cambios Priorizado

#### PRIORIDAD 1: Rotacion 2D Real

- [ ] **P1.1** Crear funcion `rotate_points(points, center, angle_rad)` que aplique transformacion 2D
- [ ] **P1.2** Modificar `_draw_single_figure()` para rotar vertices antes de dibujar
- [ ] **P1.3** Para circulos: rotar un punto de referencia visible (marca de rotacion)
- [ ] **P1.4** Asegurar que la rotacion usa el centroide de la figura como pivot

**Implementacion propuesta:**
```python
def rotate_points(points: np.ndarray, center: tuple, angle_rad: float) -> np.ndarray:
    """Rota puntos 2D alrededor de un centro."""
    cx, cy = center
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    # Trasladar al origen, rotar, trasladar de vuelta
    translated = points - np.array([cx, cy])
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = translated @ rotation_matrix.T
    return (rotated + np.array([cx, cy])).astype(np.int32)
```

---

#### PRIORIDAD 2: Rotacion Continua con Mano Abierta

- [ ] **P2.1** Cambiar de incremento fijo (+6°) a velocidad basada en tiempo
- [ ] **P2.2** Mientras `OPEN_HAND` activo: `rotation += rotation_speed * dt`
- [ ] **P2.3** Al cerrar mano (cambio de gesto): congelar rotacion
- [ ] **P2.4** Variable `rotation_speed` configurable (default: 90°/segundo)

**Implementacion propuesta:**
```python
# En handle_gestures()
if gesture == Gesture.OPEN_HAND and self.selected_figure:
    dt = current_time - self.last_frame_time
    self.selected_figure['rotation'] += self.rotation_speed * dt
    self.selected_figure['rotation'] %= 360
# Al cambiar de OPEN_HAND a otro gesto: no hacer nada (ya esta congelado)
```

---

#### PRIORIDAD 3: Separacion Pinch/Rotacion

- [ ] **P3.1** Agregar estado `is_rotating` booleano
- [ ] **P3.2** PINCH solo activo si `is_rotating == False`
- [ ] **P3.3** OPEN_HAND solo rota si no hay pinch activo
- [ ] **P3.4** Mutex implicito: un gesto a la vez

**Estado actual:** Ya parcialmente implementado (pinch tiene prioridad en linea 193)

---

#### PRIORIDAD 4: Color Configurable desde Teclado

**Recomendacion: Presets con Teclas Numericas + Indicador Visual**

- [ ] **P4.1** Teclas 0-9 para presets de color (10 colores predefinidos)
- [ ] **P4.2** Teclas R/G/B para ajustar canal activo con +/-
- [ ] **P4.3** Tecla TAB para ciclar canal activo (R→G→B→R)
- [ ] **P4.4** Mostrar color actual en HUD con valores RGB
- [ ] **P4.5** Guardar ultimo color usado en config

**Implementacion propuesta:**
```python
# Presets (mas estable que sliders en CLI)
COLOR_PRESETS = [
    (255, 0, 0),    # 0: Rojo
    (0, 255, 0),    # 1: Verde
    (0, 0, 255),    # 2: Azul
    (255, 255, 0),  # 3: Amarillo
    (255, 0, 255),  # 4: Magenta
    (0, 255, 255),  # 5: Cyan
    (255, 128, 0),  # 6: Naranja
    (128, 0, 255),  # 7: Purpura
    (255, 255, 255),# 8: Blanco
    (128, 128, 128) # 9: Gris
]

# Ajuste fino con R/G/B + flechas
self.active_channel = 0  # 0=R, 1=G, 2=B
# Tecla arriba: color[active_channel] += 10
# Tecla abajo: color[active_channel] -= 10
```

---

#### PRIORIDAD 5: Suavizado de Landmark

**Recomendacion: EMA (Exponential Moving Average)**

- [ ] **P5.1** Crear clase `EMAFilter` con parametro `alpha` configurable
- [ ] **P5.2** Aplicar a `current_pos` en ColorPainter ANTES de dibujar
- [ ] **P5.3** Aplicar a `pinch_position` en Gesture3D
- [ ] **P5.4** Parametro `alpha` entre 0.3-0.5 (balance jitter/latencia)

**Implementacion propuesta:**
```python
class EMAFilter:
    """Filtro Exponential Moving Average para suavizar coordenadas."""

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha  # 0 < alpha <= 1 (mayor = menos suavizado)
        self.value = None

    def update(self, new_value: tuple) -> tuple:
        if self.value is None:
            self.value = new_value
        else:
            self.value = (
                self.alpha * new_value[0] + (1 - self.alpha) * self.value[0],
                self.alpha * new_value[1] + (1 - self.alpha) * self.value[1]
            )
        return (int(self.value[0]), int(self.value[1]))

    def reset(self):
        self.value = None
```

**Alternativa Kalman (mas complejo pero mejor para movimiento predecible):**
```python
# Solo si EMA no es suficiente
# Usar cv2.KalmanFilter con 4 estados (x, y, vx, vy)
```

---

#### PRIORIDAD 6: Tests

- [ ] **P6.1** Configurar pytest con conftest.py
- [ ] **P6.2** Fixtures para frames simulados (imagen numpy con forma de mano)
- [ ] **P6.3** Mocks para MediaPipe (hand_landmarks predefinidos)
- [ ] **P6.4** Tests unitarios para EMAFilter
- [ ] **P6.5** Tests unitarios para rotate_points()
- [ ] **P6.6** Tests de integracion para deteccion de gestos
- [ ] **P6.7** Tests de regresion para color detection

**Estructura de tests:**
```python
# test_filters.py
def test_ema_filter_reduces_jitter():
    filter = EMAFilter(alpha=0.3)
    noisy_points = [(100, 100), (102, 98), (99, 101), (101, 99)]
    smoothed = [filter.update(p) for p in noisy_points]
    # Verificar que varianza disminuye

# test_gesture_detector.py
def test_open_hand_detected_with_5_fingers():
    mock_landmarks = create_mock_open_hand()
    detector = GestureDetector()
    gesture = detector.detect(mock_landmarks)
    assert gesture == Gesture.OPEN_HAND

# test_figures.py
def test_square_rotation_90_degrees():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    center = (5, 5)
    rotated = rotate_points(np.array(square), center, np.pi/2)
    expected = [(10, 0), (10, 10), (0, 10), (0, 0)]
    np.testing.assert_array_almost_equal(rotated, expected)
```

---

### 4.3 Archivos a Modificar (Proyecto 1)

| Archivo Actual | Accion | Archivo Destino |
|----------------|--------|-----------------|
| `main.py` | Refactorizar | `proyecto1/src/main.py`, `proyecto1/src/core/app.py` |
| `Gesture3D.py` | Refactorizar + Fix rotacion | `proyecto1/src/tracking/gesture_detector.py`, `proyecto1/src/drawing/figures.py` |
| `ColorPainter.py` | Refactorizar + Agregar filtro | `proyecto1/src/tracking/color_detector.py`, `proyecto1/src/drawing/canvas.py` |
| `neon_effects.py` | Limpiar/Integrar | `proyecto1/src/drawing/effects.py` |
| N/A | Crear | `proyecto1/src/tracking/filters.py` |
| N/A | Crear | `proyecto1/tests/*.py` |
| N/A | Crear | `proyecto1/README.md` |

---

## 5. PLAN DE CAMBIOS - PROYECTO 2 (Alto Nivel)

### 5.1 Objetivo

Filtro tipo Snapchat neon futurista con:
- FaceMesh para tracking facial
- OpenGL + GLFW para renderizado 3D
- Escalado proporcional al rostro
- Animacion por expresion facial
- Profundidad falsa (parallax)
- Grabacion de video

### 5.2 Estructura Propuesta

```
proyecto2/
├── README.md
├── requirements.txt          # + PyOpenGL, glfw
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── face/
│   │   ├── __init__.py
│   │   ├── mesh_detector.py     # MediaPipe FaceMesh
│   │   ├── expression_analyzer.py# Detectar sonrisa, cejas, etc.
│   │   └── landmarks.py         # Indices clave del mesh
│   ├── rendering/
│   │   ├── __init__.py
│   │   ├── gl_context.py        # Inicializacion OpenGL/GLFW
│   │   ├── shaders.py           # Vertex/Fragment shaders neon
│   │   ├── filter_renderer.py   # Render del filtro sobre cara
│   │   └── depth_simulator.py   # Efecto parallax pseudo-3D
│   ├── effects/
│   │   ├── __init__.py
│   │   ├── neon_glow.py         # Efecto glow shader
│   │   ├── cyberpunk_overlay.py # Overlays futuristas
│   │   └── animations.py        # Animaciones por expresion
│   └── recording/
│       ├── __init__.py
│       └── video_recorder.py    # cv2.VideoWriter wrapper
└── assets/
    ├── shaders/
    │   ├── neon.vert
    │   └── neon.frag
    └── textures/
        └── glow_mask.png
```

### 5.3 Checklist Alto Nivel

- [ ] **Setup OpenGL/GLFW** con contexto compartido para overlay
- [ ] **Integrar FaceMesh** (478 landmarks) con indices clave
- [ ] **Calcular bounding box facial** para escalar filtro
- [ ] **Shaders neon** con glow dinamico
- [ ] **Detector de expresiones** (sonrisa, cejas levantadas, boca abierta)
- [ ] **Animar filtro** segun expresion (ej: glow mas intenso al sonreir)
- [ ] **Profundidad falsa** con parallax basado en rotacion de cabeza
- [ ] **VideoRecorder** con codec configurable (mp4/webm)
- [ ] **README.md** con instrucciones y demo

### 5.4 Dependencias Adicionales

```
PyOpenGL==3.1.7
glfw==2.6.3
Pillow==10.2.0  # Para texturas
```

---

## 6. RIESGOS TECNICOS Y MITIGACIONES

### 6.1 Riesgos Proyecto 1

| Riesgo | Probabilidad | Impacto | Mitigacion |
|--------|--------------|---------|------------|
| **Rendimiento degradado con rotacion** | Media | Alto | Pre-calcular vertices rotados, cache de figuras estaticas |
| **EMA introduce latencia perceptible** | Media | Medio | Parametro alpha configurable, mostrar ambos (raw/smooth) en debug |
| **Conflicto pinch/rotacion** | Baja | Medio | Maquina de estados explicita para gestos |
| **Tests con MediaPipe complejo** | Alta | Medio | Mocks de landmarks, no depender de modelo real en unit tests |
| **Colores no visibles en ciertos fondos** | Media | Bajo | Agregar outline/contraste automatico |

### 6.2 Riesgos Proyecto 2

| Riesgo | Probabilidad | Impacto | Mitigacion |
|--------|--------------|---------|------------|
| **Compatibilidad OpenGL en Linux** | Media | Alto | Probar con Mesa drivers, fallback a software rendering |
| **Sincronizacion OpenGL+OpenCV** | Alta | Alto | Usar texturas compartidas o FBO→numpy explicito |
| **FaceMesh pesado (478 landmarks)** | Media | Medio | Usar solo landmarks necesarios, downscale frame |
| **Grabacion causa frame drops** | Alta | Medio | Thread separado para encoding, buffer circular |
| **Shaders no portables** | Media | Medio | GLSL version minima (3.30), evitar extensiones |

### 6.3 Mitigaciones Generales

1. **Profiling continuo:** Medir FPS en cada cambio significativo
2. **Feature flags:** Poder deshabilitar features nuevas rapidamente
3. **Logging estructurado:** Para debugging de estados de gestos
4. **Configuracion externalizada:** No hardcodear umbrales/colores
5. **Graceful degradation:** Si MediaPipe falla, mostrar frame sin procesar

---

## 7. RESUMEN EJECUTABLE

### Pasos Inmediatos (Proyecto 1)

1. **Crear estructura de carpetas** `proyecto1/src/`, `proyecto1/tests/`
2. **Mover y renombrar archivos** segun seccion 4.3
3. **Implementar `rotate_points()`** y aplicar en `_draw_single_figure()`
4. **Cambiar logica de rotacion** a continua mientras OPEN_HAND
5. **Implementar `EMAFilter`** y aplicar en color detector y gesture detector
6. **Agregar teclas de color** 0-9 para presets
7. **Escribir tests** para filtros y rotacion
8. **Escribir README.md** con instrucciones de uso

### Pasos Inmediatos (Proyecto 2)

1. **Crear estructura de carpetas** `proyecto2/src/`
2. **Setup basico OpenGL+GLFW** con ventana de prueba
3. **Integrar FaceMesh** y mostrar landmarks
4. **Calcular bounding box** y escalar overlay
5. **Implementar shader neon basico**
6. **Agregar deteccion de expresiones**
7. **Implementar grabacion de video**

---

## 8. ARCHIVOS QUE SE MODIFICARIAN

### Proyecto 1

| Archivo | Tipo de Cambio |
|---------|----------------|
| `ColorPainter.py` | Agregar EMAFilter, refactorizar a modulo |
| `Gesture3D.py` | Fix rotacion real, separar figura/detector, agregar EMA |
| `main.py` | Refactorizar, agregar controles de color |
| `neon_effects.py` | Integrar o eliminar (codigo muerto) |
| `requirements.txt` | Agregar pytest |
| **NUEVOS:** | `filters.py`, `figures.py`, `config.py`, `tests/*.py`, `README.md` |

### Proyecto 2

| Archivo | Tipo de Cambio |
|---------|----------------|
| **NUEVOS:** | Todo el proyecto desde cero |
| `requirements.txt` | PyOpenGL, glfw, Pillow |

---

*Fin del Analisis - Documento generado automaticamente*
