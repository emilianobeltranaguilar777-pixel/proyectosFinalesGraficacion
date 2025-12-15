# ANALISIS COMPLETO DEL REPOSITORIO
## Neural Canvas v2.1 - Aplicacion de arte interactivo por vision por computadora

**Fecha de analisis:** 2025-12-15  \
**Contexto:** Ubuntu/Linux | numpy==1.26.4, opencv-python==4.8.1.78, mediapipe==0.10.21, pytest>=7.0.0

---

## 1. MAPA DEL REPOSITORIO

### 1.1 Estructura general
```
/proyectosFinalesGraficacion
├── main.py             # Entrypoint y orquestador de modos (657 lineas)
├── Gesture3D.py        # Gestos de mano + figuras geometricas (897 lineas)
├── ColorPainter.py     # Seguimiento de color y pintura libre (426 lineas)
├── neon_menu.py        # Menu radial estable renderizado en OpenCV (296 lineas)
├── geometry_utils.py   # Utilidades geometricas (37 lineas)
├── filters.py          # Filtro EMA para suavizado (71 lineas)
├── neon_effects.py     # Efectos visuales opcionales (165 lineas)
├── requirements.txt    # Dependencias (OpenCV, MediaPipe, numpy, pytest)
└── tests/              # Suite Pytest de rotacion, escalado, menu y tracking
```
**Total estimado:** 2,549 lineas de Python activas.

### 1.2 Test suite
- 9 archivos de prueba unitarios/integracion en `tests/` cubren rotacion continua y mutex con pinch, escalado angular, spawn seguro, smoothing, menu neon y tracking de color. Fixtures en `conftest.py` simulan figuras, frames y objetos de Gesture3D/ColorPainter.

---

## 2. DESCRIPCION DE MODULOS PRINCIPALES

### 2.1 `main.py` – controlador principal
- Clase **`PizarraNeon`** inicializa captura 1280x720@30fps, ColorPainter, Gesture3D (modo menu externo) y el menu radial `NeonMenu`. Administra modo actual (`menu`, `color`, `gestos`), cache de grid, paleta de colores y un **FPSCounter** de ventana movil.
- HUD minimalista: textos con sombra, grid cacheado y barra superior; desactiva efectos pesados si los FPS caen por debajo de 15.
- Menu radial: callbacks crean figuras con posicion segura (clamping a bordes) y cierran el menu en el mismo frame. Incluye boton de borrado.
- Bucle principal (`ejecutar`): captura frame, espeja, delega a modo color o gestos, actualiza/dibuja menu, muestra HUD y procesa teclas (`1` modo color, `2` modo gestos, `M` menu, `Q` salir, `D` perf HUD).

### 2.2 `ColorPainter.py` – tracking de color y pintura
- Configura canvas RGB con pincel ajustable, particulas de trazo y paleta de 8 colores. Usa **EMAFilter** para suavizar el centroide del objeto azul detectado.
- `detect_blue_object`: convierte a HSV, umbral fijo [100,120,50]-[130,255,255], morfologia, elige mayor contorno y calcula centroide por momentos.
- `process_frame`: detecta centroide, suaviza coordenadas, dibuja lineas entre last_pos y current_pos, renderiza particulas y panel de UI inferior (color, grosor, FPS del modo).
- Atajos: `SPACE` limpia canvas, `C` cambia color ciclico, `+/-` ajustan grosor; modo color en `main.py` overlaya canvas con alpha.

### 2.3 `Gesture3D.py` – figuras geometricas + gestos MediaPipe
- Inicializa MediaPipe Hands (1 mano, confidencia 0.7) de forma tolerante a errores; si no hay dependencias, mantiene interfaz sin gestos.
- Gestos soportados: **PINCH** (agarre/escala), **VICTORY** (toggle menu radial), **OPEN_HAND** (rotacion continua), **FIST/NONE** (reposo). Gestos se derivan de dedos extendidos calculados desde landmarks en pixeles.
- **Rotacion real y continua**: `rotate_figure_continuous(dt)` suma `rotation_speed` (pi rad/s por defecto) mientras OPEN_HAND esta activo; mutex evita rotar cuando hay PINCH. `_draw_single_figure` rota vertices usando `rotate_points` para cuadrados, triangulos, estrellas, corazones y hexagonos.
- **Escalado doble**: modo `NORMAL` escala por distancia pulgar-indice; modo `SCALE` habilita **escala angular** (giro alrededor del centro suma/tambien resta tamaño con factor -40 px/rad) con limites 15-300 px.
- **Menu de figuras**: puede ser interno (radial de 6 figuras + color/delete) o externo via `NeonMenu`. Muestra estado `menu_state`, cooldown para gesto VICTORY y toggle de seleccion.
- **Suavizado de pinch**: `EMAFilter` en `pinch_filter` mantiene coordenadas estables al mover figuras. Guarda `last_pinch_position` para spawn seguro.
- Helpers: borrado de figuras (`delete_selected_figure`, `clear_figures`), cambio de color ciclico, dibujos HUD (cursor, centroide, menu), y estadisticas FPS por etapa.

### 2.4 `neon_menu.py` – menu radial estable
- Menu deterministico sin glow/pulse: calcula posiciones cada frame, animacion lineal de apertura/cierre y hover lineal. `MenuButton` define etiqueta, color y callback `on_select`.
- Soporta hit-test radial con zona muerta interna y escala por `animation`. Dibuja circulo base, botones con borde y iconos vectoriales simples (circle, square, triangle, star, heart, hexagon, delete).

### 2.5 Utilidades
- `geometry_utils.py`: `rotate_points(points, center, angle_rad)` rota arrays (N,2) alrededor de un centro con matriz de rotacion; retorna `np.int32`.
- `filters.py`: `EMAFilter(alpha=0.4)` suaviza coordenadas 2D, mantiene ultimo valor cuando no hay deteccion y expone `reset`, `has_value`, `raw_value`.
- `neon_effects.py`: efectos opcionales (texto glow, grid, matrix rain, bordes, scanlines, patrones hexagonales). No se invoca en flujo principal actual.

### 2.6 `tests/` – cobertura principal
- **Rotacion y mutex** (`test_rotation.py`): verifica rotacion continua mientras OPEN_HAND, congelamiento al soltar, velocidad proporcional a dt y ausencia de rotacion durante PINCH.
- **Escalado angular** (`test_angular_scale.py`): confirma incremento/decremento de tamaño por angulo y limites min/max.
- **Spawn y seleccion** (`test_spawn_position.py`): comprueba clamping de spawn seguro y preservacion de figuras tras borrar seleccion.
- **Menu neon** (`test_neon_menu*.py`): hit-test radial, animacion lineal y callbacks de botones.
- **Filtros/geom** (`test_filters.py`, `test_geometry_utils.py`): EMA mantiene suavizado y `rotate_points` respeta matrices.
- **Tracking color** (`test_color_tracking.py`): construye mascara azul y valida centroide en un frame sintetico.

---

## 3. COMPORTAMIENTO DEL SISTEMA
- **Modos**: `menu` (pantalla de seleccion), `color` (pintura por color) y `gestos` (figuras controladas con mano). Teclas en `main.py` cambian modo y activan HUD de desempeño.
- **Input gestual**: PINCH prioriza sobre OPEN_HAND; se limpia al perder gesto. VICTORY alterna visibilidad del menu radial con cooldown de 0.5s para evitar rebotes.
- **Pintura**: Canvas persistente sobre frame espejado; particulas de trazo con fading; overlay alpha configurable. Brush size limitado [3,60].
- **Performance**: contador FPS de ventana movil; metricas por etapa (deteccion, handle, draw, menu). Modo low FPS deshabilita efectos visuales adicionales.
- **Resiliencia**: manejo defensivo cuando MediaPipe u OpenCV no estan disponibles (import seguro en Gesture3D/neon_menu); filtros EMA retienen ultima posicion para evitar saltos al perder deteccion momentaneamente.

---

## 4. OBSERVACIONES Y OPORTUNIDADES
- El analisis previo ya esta parcialmente implementado: rotacion real, suavizado EMA y mutex pinch/rotacion estan en produccion y probados por Pytest.
- Pending: ajustes de paletas configurables desde teclado y documentacion de usuario final; `neon_effects.py` sigue sin integrarse en el flujo.
- `requirements.txt` carece de salto de linea final; instalar `pytest` es requerido para la suite local.

---

*Documento actualizado automaticamente tras inspeccion completa del repositorio.*
