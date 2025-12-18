# ANALISIS COMPLETO DEL REPOSITORIO
## Neural Canvas v2.1 - “Informe literal exhaustivo del código”

**Fecha de Análisis:** 2025-12-14  
**Contexto:** Ubuntu/Linux, numpy==1.26.4, opencv-python==4.8.1.78, mediapipe==0.10.21

---

## A) Árbol del proyecto y rol de cada archivo

```
/proyectosFinalesGraficacion
├── ANALISIS_COMPLETO.md      # Documento de análisis (este archivo)
├── main.py                   # ENTRYPOINT principal (loop OpenCV + modos)
├── ColorPainter.py           # Tracking de color + pintura y partículas 2D
├── Gesture3D.py              # Gestos MediaPipe para figuras 2D + menú externo
├── neon_menu.py              # Menú radial NeonMenu (UI OpenCV)
├── neon_effects.py           # Biblioteca de efectos neon (no invocada en main)
├── geometry_utils.py         # Utilidades geométricas (rotación 2D)
├── filters.py                # Filtros temporales (EMAFilter)
├── ar_filter/
│   ├── __init__.py           # Módulo vacío
│   ├── gl_app.py             # ENTRYPOINT AR OpenGL/GLFW
│   ├── face_tracker.py       # Wrapper MediaPipe FaceMesh (datos puros)
│   ├── metrics.py            # Métricas de landmarks (pura matemática)
│   ├── primitives.py         # Generación de geometría (numpy)
│   └── shaders/              # Shaders GLSL (basic.vert/frag, particle.vert/frag)
├── requirements.txt          # Dependencias
└── tests/                    # Pruebas unitarias e integración
```

**Entrypoints explícitos:**
- `main.py` ejecuta la aplicación de cámara principal (`PizarraNeon.ejecutar`).
- `ar_filter/gl_app.py` expone `run_ar_filter` para lanzar el filtro AR OpenGL.

---

## B) Flujo runtime end-to-end (narrativa por etapas)

1. **Inicio del programa:** `python main.py` crea `PizarraNeon` y llama `ejecutar()` (`main.py:868-899`).
2. **Inicialización de cámara / MediaPipe:** `inicializar()` abre `cv2.VideoCapture(0)` y configura 1280×720 @30fps (`main.py:61-99`). `Gesture3D.__init__` llama `_initialize_mediapipe` para `mp.solutions.hands.Hands` (`Gesture3D.py:27-110`).
3. **Inicialización OpenGL/GLFW (solo modo AR):** desde menú principal, tecla `3` llama `_lanzar_ar_filter` → `ar_filter.gl_app.run_ar_filter` crea ventana GLFW, shaders y geometría (`main.py:681-733`, `gl_app.py:106-471`).
4. **Inicialización del sistema de menú (NeonMenu):** `PizarraNeon._crear_neon_menu` instancia `NeonMenu` con botones basados en `Gesture3D.available_figures` (`main.py:31-59`). `NeonMenu` arranca en estado `HIDDEN` con animación controlada (`neon_menu.py:24-147`).
5. **Loop principal de render (OpenCV):** dentro de `ejecutar()`, `while True` captura frame, espejo horizontal, redimensiona y despacha por modo (`main.py:868-914`).
6. **Lectura de frame → extracción de landmarks → métricas:**
   - **Modo color:** `ColorPainter.process_frame` detecta objeto por HSV y filtra centroide con EMA (`ColorPainter.py:205-282`).
   - **Modo gestos:** `Gesture3D.process_frame` llama `detect_gestures` (MediaPipe Hands) para `Gesture` + `pinch_position` (`Gesture3D.py:618-655`).
   - **Modo AR:** `FaceTracker.process_frame` obtiene landmarks FaceMesh normalizados (`face_tracker.py:32-87`); métricas puras en `metrics.py` calculan anchuras, centros y aperturas.
7. **Render overlay / primitivas / partículas:**
   - OpenCV: grids, HUDs, menús, figuras 2D (`main.py`, `Gesture3D.draw_interface`, `ColorPainter.draw_paint_effects`).
   - OpenGL: fondo con textura de cámara, halo de cubos con estelas, boca/ojos robóticos (`gl_app.py:473-1179`).
8. **Manejo de inputs (teclas/gestos):** `procesar_teclas` conmuta modos y toggles (`main.py:604-675`); `Gesture3D.handle_gestures` decide selección, escala, rotación continua y toggles de menú externo (`Gesture3D.py:213-258`). GLFW captura teclas A/R/V/B/D/ESC para color de tracking y debug (`gl_app.py:176-220`).
9. **Limpieza y shutdown:** `liberar_recursos` libera `VideoCapture` y destruye ventanas; `_cleanup` en AR borra VAOs/VBOs, programas y llama `glfw.terminate()` (`main.py:921-935`, `gl_app.py:1227-1257`).

**Loop por frame (pseudocódigo obligatorio):**
```
while running:
  read_frame()
  landmarks = facemesh(frame)            # Hands en Gesture3D o FaceMesh en AR
  metrics = compute_metrics(landmarks)   # mouth/eyes/halo/scale según módulo
  handle_inputs()
  update_menu_state()
  update_particles()
  render_face_overlays()
  swap_buffers()
```

---

## 2) Informe literal por archivo (línea por línea “conceptual”)

### main.py (ENTRYPOINT OpenCV)

**Resumen:** Orquesta modos “menu”, “color”, “gestos”, lanza AR externo y gestiona HUD/overlays. Importa `ColorPainter` para tracking, `Gesture3D` para gestos 3D, `NeonMenu` para UI radial y fija `QT_QPA_PLATFORM` (`main.py:1-20`).

**Clases/funciones expuestas:**
- `class PizarraNeon`: estado global del bucle principal (`main.py:22-936`).
- `PizarraNeon.ejecutar`: loop infinito de captura y render (`main.py:868-914`).

**Detalles por método/estado:**
- `__init__`: define cámara (`cap=None`), modo actual, resolución 1280×720, caches de grid, contadores FPS, paleta de colores y parámetros de zonas de escala con `scale_lock_active` (`main.py:22-114`).
- `_crear_neon_menu`: arma botones `MenuButton` con callbacks que crean figuras o borran la seleccionada, define radio, deadzone y colores (`main.py:31-59`).
- `inicializar`: configura `cv2.VideoCapture`, parámetros y mensajes de consola (`main.py:61-106`).
- `_posicion_segura_creacion`: limita spawn a márgenes de 80 px alrededor de la preferencia de pinch/cursor (`main.py:108-126`).
- `dibujar_texto_limpio`: doble `putText` con sombra para HUD (`main.py:128-138`).
- `dibujar_grid_minimal`: cachea grid cada 0.2s, usa `sin` para pulso y `addWeighted` para mezclar (`main.py:140-185`).
- `dibujar_borde_esquinas`: anima esquinas con color pulsante basado en `sin` (`main.py:187-238`).
- `dibujar_display_seleccion`: renderiza header verde con info de figura seleccionada, miniaturas según tipo (`main.py:240-329`).
- `dibujar_zonas_escala`: cuando `SelectionMode.SCALE` y hay figura, pinta rectángulos semi-transparentes izquierda/derecha y labels “- SIZE/+ SIZE” (`main.py:331-400`).
- `_calcular_scale_lock`: activa `scale_lock_active` si pinch activo en zona de escala dentro de márgenes verticales; bloquea otras interacciones (`main.py:402-440`).
- `_manejar_escala_bloqueada`: delega a `Gesture3D.handle_figure_scaling_by_spatial` con límites precomputados (`main.py:442-453`).
- `dibujar_menu_principal`: compone pantalla de inicio con títulos, opciones 1-3-Q y footer (`main.py:455-531`).
- `dibujar_hud_superior`: barra superior con modo actual y FPS, cambia color según FPS (>25 verde) (`main.py:533-550`).
- `_dibujar_overlay_perf`: panel opcional (tecla F) con tiempos de etapas (`main.py:552-577`).
- `modo_seguimiento_color`: prepara HUD y delega a `ColorPainter.process_frame` (`main.py:579-591`).
- `modo_figuras_gestos`: pipeline completo gestos; calcula `dt`, evalúa `scale_lock_active`, mantiene rotación bloqueada si menú activo, actualiza menú neon y pinta zonas de escala. Si `debug_perf`, guarda timings (`main.py:593-676`).
- `procesar_teclas`: switchea modos (q,1,2,3,m,f), deriva a handlers por modo (`main.py:604-675`).
- `_procesar_teclas_color`: atajos para canvas (clear, color, tamaño, presets HSV, toggle calibración) (`main.py:677-708`).
- `_procesar_teclas_gestos`: crea figuras por número, limpia, elimina, alterna `scale_mode`; cuando `scale_lock_active` sólo permite salir de escala (`main.py:710-774`).
- `_lanzar_ar_filter`: libera cámara OpenCV, llama `run_ar_filter`, y luego restaura `VideoCapture` y ventana, regresando a modo menú (`main.py:776-836`).
- `_actualizar_neon_menu`: sincroniza cursor (`ultima_pos_cursor`) con gestos, consume toggles `VICTORY`, abre/cierra menú y llama `NeonMenu.update` (`main.py:838-866`).
- `ejecutar`: loop principal con FPS counter, captura, espejo, resize, dispatch por modo y lectura de teclado (`main.py:868-914`).
- `liberar_recursos`: libera cámara y cierra ventanas (`main.py:916-935`).

**Loops clave:** `while True` en `ejecutar` recorre frames; loops de dibujo de grid recorren coordenadas X/Y espaciadas (`main.py:140-185`); menú principal itera items `menu_items` (`main.py:477-520`).

**Condicionales clave:**
- Modo activo controla pipeline (`if self.modo_actual == ...`, `main.py:885-906`).
- `scale_lock_active` divide flujo entre escala bloqueada vs normal (`main.py:616-666`).
- Entradas de teclado cambian estados (`main.py:604-675`).

**Parámetros sensibles:** `grid_update_interval=0.2`, `scale_zone_left_pct=0.33`, `scale_zone_right_pct=0.67`, `scale_zone_margin_top_pct=0.12`, `scale_zone_margin_bottom_pct=0.12`, colores de HUD en diccionario `self.colores` (`main.py:73-115`).

### Gesture3D.py (gestos y figuras 2D)

**Resumen:** Gestiona detección de mano con MediaPipe Hands, mantiene lista de figuras 2D, maneja selección, escala (dedos o zonas espaciales), movimiento y rotación continua; integra menú externo (NeonMenu) mediante toggles `VICTORY` (`Gesture3D.py:1-657`).

**Clases/estados:**
- `Gesture`, `MenuState`, `SelectionMode` enumeran gestos y modos (`Gesture3D.py:6-25`).
- `Gesture3D` mantiene `figures`, `selected_figure`, `pinch_active`, `menu_state`, `available_figures`, `selection_mode`, `_scale_zone_active`, `rotation_enabled`, `pinch_filter` (EMA) (`Gesture3D.py:27-92`).

**Funciones principales:**
- `_initialize_mediapipe`: instancia `mp.solutions.hands.Hands` con `min_detection_confidence=0.7`, tracking 0.7 y 1 mano (`Gesture3D.py:94-110`).
- `_get_pixel_landmarks`: mapea landmarks normalizados a píxeles (`Gesture3D.py:112-115`).
- `_get_finger_states`: computa dedos extendidos comparando tip/pip y lado de la mano (`Gesture3D.py:117-139`).
- `detect_gestures`: procesa frame BGR→RGB, ejecuta `hands.process`, calcula `thumb_tip`, `index_tip`, distancia entre dedos (`current_finger_distance`), umbral de pinch (`hand_width*0.3`), y clasifica en PINCH/VICTORY/FIST/OPEN_HAND (`Gesture3D.py:141-211`).
- `handle_gestures`: gestiona estados: toggles de menú externo si `use_external_menu`, selección/creación de figuras en pinch, escala por dedos en modo `SCALE`, movimiento de figura, y rotación continua con mano abierta (mutex con pinch/menú externo) (`Gesture3D.py:213-258`).
- `_update_menu_toggle_state` y `consume_menu_toggle`: registran y consumen gesto `VICTORY` para abrir/cerrar menú radial externo (`Gesture3D.py:260-282`).
- `handle_figure_scaling_by_fingers`: calcula factor `scale_reference_distance` vs `current_finger_distance`, aplica `_apply_smoothing` y suaviza tamaño con `size_smoothing=0.2`, límites `[min_figure_size=15, max_figure_size=300]` (`Gesture3D.py:284-333`).
- `_apply_smoothing`: easing diferenciado >1 o <1 (`Gesture3D.py:335-343`).
- `handle_figure_scaling_by_spatial`: escala según zona izquierda/derecha y distancia al borde, velocidad `_spatial_scale_speed=150 px/s`, activa `_scale_zone_active` (`Gesture3D.py:345-396`).
- `set_rotation_enabled`/`set_external_menu_active`: gates para evitar conflictos de rotación o entrada cuando menú externo está visible (`Gesture3D.py:414-432`).
- `handle_menu_selection`: construye items en círculo (radio 0.6*menu_radius) más botones `color/delete`; sobre pinch dentro de radio `size` ejecuta cambio de color, delete o creación de figura (`Gesture3D.py:434-474`).
- `handle_figure_selection`: selecciona figura más cercana dentro de `size*1.2`, ajusta `selection_color` según modo; si no hay, crea nueva (`Gesture3D.py:476-506`).
- `create_figure`/`create_figure_by_key`: genera dict con `type/position/size/color/rotation/selection_color/creation_time/id`, tamaño inicial 60 y selecciona la nueva figura (`Gesture3D.py:508-528`).
- `move_figure`: limita posición por margen `size+10` dentro de pantalla (`Gesture3D.py:530-539`).
- `rotate_figure_continuous`: incrementa `rotation` en radianes usando `rotation_speed=pi rad/s` y `dt` (`Gesture3D.py:545-556`).
- `delete_selected_figure`/`clear_figures`: eliminan figura actual o todas (`Gesture3D.py:558-571`).
- `toggle_scale_mode`: alterna `SelectionMode`, ajusta `selection_color` rojo o amarillo (`Gesture3D.py:573-587`).
- `draw_finger_connection_line`: visualiza línea entre pulgar e índice con color según distancia (`Gesture3D.py:589-604`).
- `draw_enhanced_menu` + `_draw_figure_menu_button` + `_draw_control_buttons`: render radial interno (cuando `use_external_menu=False`) con figuras y botones de control (`Gesture3D.py:606-705`).
- Figuras rotadas: `_draw_star_rotated`, `_draw_heart_rotated`, `_draw_hexagon_rotated` usan `rotate_points` para aplicar `angle_rad` (`Gesture3D.py:707-790`).
- `_draw_single_figure`: switch por tipo con rotación real y resaltado de selección; dibuja indicador central y línea de rotación si `angle_rad!=0` (`Gesture3D.py:792-841`).
- `draw_interface`: compone línea de dedos, pinch point, menú interno, figuras y landmarks (`Gesture3D.py:843-873`).
- `process_frame`: pipeline: detect_gestures → EMA `pinch_filter` → handle_gestures → draw_interface; llena `profile` opcional con timings (`Gesture3D.py:875-906`).

**Loops importantes:**
- Recorridos de figuras en `draw_figures` y `_draw_single_figure` (`Gesture3D.py:788-841`).
- Menú radial recorre `available_figures` para posiciones (`Gesture3D.py:606-657`).

**Condicionales de estado:**
- `selection_mode` decide entre mover y escalar (`Gesture3D.py:213-258`).
- `external_menu_active` deshabilita rotación y pinch handling normal (`Gesture3D.py:217-233`).
- `gesture == Gesture.PINCH` diferencia inicio/continuación/liberación (`Gesture3D.py:233-255`).

**Parámetros sensibles:** `gesture_cooldown=0.5s`, `min_finger_distance=20`, `max_finger_distance=250`, `_spatial_scale_speed=150`, `size_smoothing=0.2`, `rotation_speed=pi`.

### ColorPainter.py (tracking HSV + pintura)

**Resumen:** Detecta objeto por rango HSV configurable con presets, suaviza centroide con `EMAFilter`, dibuja trazos en canvas y partículas de efecto de pintura; UI con info HSV y brush (`ColorPainter.py:1-282`).

**Estado/propiedades:** `canvas` negro, `brush_size=15`, `brush_color` inicial azul eléctrico, paleta de 8 colores, `trail_particles` lista de dicts, `current_preset=1` y rangos HSV, `hsv_calibration_mode` con ventana de trackbars (`ColorPainter.py:7-73`).

**Funciones clave:**
- `set_hsv_preset`/`reset_to_default_preset`: cargan presets 1-6 (`ColorPainter.py:75-112`).
- `toggle_hsv_calibration`/`_open_hsv_calibration`/`_close_hsv_calibration`: abren/cierra trackbars y guardan valores, invalidan `last_mask` (`ColorPainter.py:114-171`).
- `_update_hsv_from_trackbars`: lee sliders en modo calibración cada frame (`ColorPainter.py:173-189`).
- `detect_color_object` (alias `detect_blue_object`): convierte a HSV, aplica inRange con límites configurables, operaciones morfológicas con kernel 3×3, cachea máscara hasta 0.1s si no calibra, retorna centroide del contorno mayor (>300 px²) (`ColorPainter.py:191-260`).
- `add_paint_effect`: emite 3 partículas por llamada con velocidad aleatoria, lifetime 0.3–0.8s, color del brush (`ColorPainter.py:262-282`).
- `update_paint_effects`: itera partículas, disminuye lifetime, integra velocidad y fricción 0.9, elimina muertas (`ColorPainter.py:284-306`).
- `draw_paint_effects`: dibuja círculos con alpha = lifetime/max_lifetime (`ColorPainter.py:308-321`).
- `draw_on_canvas`: traza líneas entre `last_pos` y `current_pos` (si distancia >2 px) con antialias, emite partículas si distancia <30; dibuja punto central y actualiza `last_pos` (`ColorPainter.py:323-351`).
- `clear_canvas`, `change_brush_color`, `change_brush_size`: reset/ciclo de color/ajuste tamaño con clamps [3,60] (`ColorPainter.py:353-373`).
- `_draw_hsv_overlay`: muestra preset y valores HSV en esquina (`ColorPainter.py:375-413`).
- `draw_ui_elements_mejorados`: panel inferior con info de brush, sample de color y crosshair animado pulsante por `sin` (`ColorPainter.py:415-475`).
- `process_frame`: actualiza partículas con dt fijo 1/30, detecta centroide (filtrado con EMA alpha=0.4), dibuja en canvas, mezcla canvas con frame (`addWeighted 0.75/0.25`), dibuja partículas y UI (`ColorPainter.py:477-528`).

**Parámetros sensibles:** presets HSV (`HSV_PRESETS`), `brush_size`, `paint_effects` toggle, `EMAFilter(alpha=0.4)`, thresholds de área (>300 px²) y cache de máscara (0.1s).

### neon_menu.py (UI radial)

**Resumen:** Menú circular animado con estados `HIDDEN/OPENING/VISIBLE/CLOSING`, botones con hover/flash, se usa como menú externo por `PizarraNeon` (`neon_menu.py:1-199`).

**Estados/propiedades:** `center`, `radius`, `inner_deadzone`, `start_angle=-pi/2`, `button_radius=20`, duraciones de apertura/cierre, `glow_intensity`, `hover_fade`, `flash_duration`, `pulse_speed`, lista de `MenuButton` con `hover_level/flash_level` (`neon_menu.py:32-87`).

**Funciones clave:**
- `open/close/is_visible`: controlan `state` y animación (`neon_menu.py:89-115`).
- `hit_test/_hit_test`: convierte cursor polar→sector para determinar botón activo; descarta si dentro de `inner_deadzone` o fuera de radio (`neon_menu.py:117-155`).
- `update`: acumula tiempo, actualiza animación (ease-out-back), calcula hover por `_hit_test`, interpola `hover_level` y decrementa `flash_level`; en flanco de `is_selecting` llama callback `on_select` del botón (`neon_menu.py:117-170`).
- `draw`: aplica escala de animación, cachea posiciones, dibuja base y botones con glow pulsante y llama `_draw_icon` según etiqueta (circle/square/triangle/otro círculo genérico) (`neon_menu.py:172-259`).
- `_update_animation`, `_get_cached_positions`, `_ease_out_back`, `_approach`, `_draw_icon`: helpers de animación, caching y dibujo de iconos (`neon_menu.py:201-259`).

**Loops:** recorre `buttons` para hover y draw; recorre `num_buttons` para calcular posiciones (`neon_menu.py:117-259`).

**Parámetros sensibles:** `open_duration=0.35`, `close_duration=0.25`, `hover_fade=0.25`, `flash_duration=0.35`, `pulse_speed=2.2`, `inner_deadzone=26`.

### geometry_utils.py (utilidad)

`rotate_points`: rota puntos 2D alrededor de centro con matriz 2×2 y devuelve `np.int32`; usado en figuras rotadas (`geometry_utils.py:1-35`).

### filters.py (EMAFilter)

`EMAFilter(alpha=0.4)`: guarda `_value` opcional, retorna valor suavizado o último conocido si input `None`; métodos `reset`, propiedades `has_value/raw_value` (`filters.py:1-57`).

### neon_effects.py

Colección de funciones de overlay (texto con glow, grids, rain, bordes, scanlines, cursor, patrón hexagonal, data stream); no se invocan en `main.py` actual (`neon_effects.py:1-166`).

### ar_filter/gl_app.py (ENTRYPOINT OpenGL)

**Resumen:** Aplicación GLFW con OpenGL 3.2 core; renderiza fondo de cámara como textura y overlays AR: halo de cubos con partículas, placa de boca robótica con barras, ojos robóticos semicirculares y opcional FaceMesh debug. Se apoya en métricas puras y geometría precacheada (`gl_app.py:1-1259`).

**Clases auxiliares:**
- `CubeData`: parámetros aleatorios de órbita/spin/wobble/color neon (`gl_app.py:47-77`).
- `Particle`: datos de partícula de estela (`pos/vel/life/color`) (`gl_app.py:79-90`).
- `OrbitingCubesSystem`: mantiene 12 cubos (`NUM_CUBES`), 500 partículas (`PARTICLE_COUNT`), tiempo y emisor circular. `update` calcula órbitas y emite 2 partículas por cubo por frame; `get_cube_transforms` devuelve posiciones/rotaciones; `get_particle_data` empaqueta datos para VBO (`gl_app.py:92-206`).

**Estado principal (`ARFilterApp`):** parámetros de ventana, contadores FPS, VAOs/VBOs para esfera, quad, cubo, semicircle, partículas; shaders para objetos, fondo, FaceMesh debug y partículas; seguimiento de color para debug mesh; smoothed values de boca/ojos (`gl_app.py:208-331`).

**Inicialización:**
- `_init_glfw`: hints OpenGL 3.2 core, crea ventana, setea callback de teclas (ESC cierra, D toggle debug mesh, A/R/V/B cambian `tracking_color`) (`gl_app.py:333-383`).
- `_init_camera`: abre `VideoCapture` y setea resolución/FPS/buffer (`gl_app.py:385-403`).
- `_load_shaders`: compila shaders desde `shaders/` y embebidos para fondo/FaceMesh (`gl_app.py:405-482`).
- `_init_geometry`: construye VAOs/VBOs para esfera/quad/cubo/semicírculo, VBO dinámico para FaceMesh (478 landmarks) y partículas; configura atributos y texturas (`gl_app.py:484-761`).

**Render helpers:**
- Matrices ortográficas/view/model (`_create_projection_matrix/_create_view_matrix/_create_model_matrix`) (`gl_app.py:763-811`).
- `_update_background_texture`: actualiza textura con frame BGR→RGB espejado (`gl_app.py:813-827`).
- `_render_background`: dibuja quad fullscreen con shader de fondo (`gl_app.py:829-843`).
- `_render_sphere`, `_render_quad`, `_render_cube`, `_render_semicircle`: setean uniforms (model/view/proj, color, lightDir, ambient) y dibujan geometría precacheada (`gl_app.py:845-1019`).
- `_render_particles`: actualiza VBO de partículas, habilita blending, usa `glDrawArrays(GL_POINTS)` con `OrbitingCubesSystem.PARTICLE_COUNT` (`gl_app.py:1021-1052`).
- `_render_facemesh_debug`: dibuja puntos y paths (oval, labios, ojos, cejas, nariz) con tracking_color, reusando VBO dinámico y `GL_LINE_STRIP/GL_POINTS` (`gl_app.py:1054-1127`).
- `_draw_landmark_path`: helper para paths (`gl_app.py:1129-1145`).

**AR overlays:**
- `_render_halo`: usa `face_width`, `halo_radius`, `forehead_center`; invierte X para espejo; actualiza `OrbitingCubesSystem`, renderiza partículas primero y luego cubos con transformaciones individuales (`gl_app.py:1147-1186`).
- `_render_mouth_rects`: calcula `mouth_openness`, suaviza `smoothed_mouth` con `smooth_value(alpha=0.2)`, obtiene `mouth_center`, `mouth_plate_dimensions`, `estimate_face_roll`; renderiza placa oscura, borde glow, barras animadas `robot_mouth_bar_heights` (7 barras) con color `robot_mouth_bar_color`; blending habilitado (`gl_app.py:1188-1292`).
- `_render_robot_eyes`: similar a boca; usa `left/right_eye_openness`, suaviza con factor 0.2, calcula radios `eye_width*1.25`, placa semicircular (`build_semicircle`), barras `robot_eye_bar_heights` (5 barras) con color `robot_eye_bar_color`, con pulse nervioso; blending habilitado (`gl_app.py:1294-1397`).

**Loop `run`:** inicializa GLFW/cámara/shaders/geom, crea `FaceTracker`, habilita depth test; imprime ayudas; en `while running` calcula `dt`, FPS, captura frame, procesa landmarks, limpia buffers, dibuja fondo, opcional debug FaceMesh sin depth, luego halo/boca/ojos si hay landmarks, swap buffers y poll events. Cierra con `_cleanup` liberando recursos GL y cámara (`gl_app.py:1399-1257`).

**Condicionales:** toggles de debug mesh (`show_debug_mesh`), color de tracking por teclas, render AR solo si `landmarks` no es None, `OPENGL_AVAILABLE` gatea constructor (`gl_app.py:25-43, 1399-1459`).

**Parámetros sensibles:** `NUM_HALO_SPHERES=8`, `SPHERE_SIZE=0.015`, `HALO_ROTATION_SPEED=0.8`, `NUM_MOUTH_RECTS=3` (barras usan 7), `RECT_BASE_WIDTH=0.02`, `RECT_BASE_HEIGHT=0.015`, `OrbitingCubesSystem.CUBE_SIZE=0.017`, `PARTICLE_COUNT=500`, `robot_mouth` smoothing alpha=0.2, `robot_eye` pulse 15Hz.

### ar_filter/face_tracker.py

`FaceTracker` encapsula `mp.solutions.face_mesh.FaceMesh` con `refine_landmarks=True`, 1 rostro, confidencias 0.5; `process_frame` convierte a RGB, procesa y retorna lista de 468/478 landmarks normalizados (caché `_last_landmarks` si no detecta) (`face_tracker.py:1-87`). `landmarks_to_screen` convierte a píxeles dados ancho/alto (`face_tracker.py:90-124`).

### ar_filter/metrics.py (módulo de métricas)

**Funciones listadas y propósito:**
- `clamp`, `lerp`, `smooth_value`: utilidades generales (`metrics.py:9-72`).
- `face_width`: distancia entre mejillas (234,454) (`metrics.py:74-104`).
- `halo_radius`: escala clamped 0.05–0.5 (`metrics.py:106-118`).
- `mouth_openness`: normaliza distancia labial (13,14) en rango 0.01–0.06 → [0,1] (`metrics.py:120-158`).
- `face_center`: retorna landmark nariz 1 o fallback (0.5,0.5,0) (`metrics.py:160-179`).
- `halo_sphere_positions` y `halo_sphere_positions_v2`: posiciones de esferas en halo elíptico (v2 usa frente y offset) (`metrics.py:181-235`).
- `mouth_rect_scale`, `mouth_rect_color`: escala y color interpolado por apertura (`metrics.py:237-270`).
- `forehead_center`: landmark 10 (top) (`metrics.py:272-290`).
- `mouth_center`: promedio de labios y comisuras (13,14,61,291) (`metrics.py:292-325`).
- `mouth_width`: distancia 61-291 (`metrics.py:327-349`).
- `robot_mouth_bar_color`: gradiente azul→amarillo→rojo según apertura (<0.5 y >=0.5) (`metrics.py:371-408`).
- `robot_mouth_bar_heights`: alturas con `lerp(base_height,max_height,openness)` + pulso `sin(time*pulse_freq+phase)` escalado por `openness` (clamp 0..max_height) (`metrics.py:410-453`).
- `estimate_face_yaw`: offset nariz vs centro mejillas → ángulo ±45° clamped (`metrics.py:455-498`).
- `estimate_face_roll`: atan2 entre ojos (33,263) (`metrics.py:500-526`).
- `mouth_plate_dimensions`: ancho=mouth_width*width_scale, alto=width*height_ratio (`metrics.py:528-548`).
- `cube_orbit_positions`: calcula órbitas de cubos con wobble y spin por `cube_data` (`metrics.py:552-596`).
- `eye_openness`: normaliza distancia párpados/anchura ojo (rango 0.05–0.25) (`metrics.py:602-646`).
- `left_eye_openness/right_eye_openness`: wrappers con índices 159/145 y 386/374 (`metrics.py:648-675`).
- `eye_center` y wrappers `left_eye_center/right_eye_center`: promedio de esquinas y párpados (`metrics.py:677-714`).
- `eye_width`, `left_eye_width/right_eye_width`: distancia entre esquinas (`metrics.py:716-748`).
- `robot_eye_bar_color`: gradiente de 4 fases (azul oscuro→cyan→amarillo→naranja) según apertura (`metrics.py:750-790`).
- `robot_eye_bar_heights`: alturas con pulse rápido 15 Hz dependiente de apertura (`metrics.py:792-829`).
- `eye_plate_dimensions`: devuelve (ancho, alto) semi-círculo basado en `eye_width*width_scale`, alto = ancho*0.5 (`metrics.py:831-850`).

**Uso en gl_app.py:**
- Halo: `face_width`, `halo_radius`, `forehead_center` (`gl_app.py:1147-1186`).
- Boca robótica: `mouth_openness`, `mouth_center`, `mouth_plate_dimensions`, `robot_mouth_bar_color`, `robot_mouth_bar_heights`, `estimate_face_roll` para animación y geometría (`gl_app.py:1188-1292`).
- Ojos robóticos: `left/right_eye_openness`, `left/right_eye_center`, `left/right_eye_width`, `robot_eye_bar_color`, `robot_eye_bar_heights`, `eye_plate_dimensions` (`gl_app.py:1294-1397`).

### ar_filter/primitives.py (geometría)

- `build_sphere`: genera vertices/normals/índices lat-long, loops en latitudes/longitudes y triángulos (`primitives.py:9-68`).
- `build_quad`: quad centrado con normales +Z (`primitives.py:70-112`).
- `build_icosphere`: icosaedro subdividido con cache de midpoint, normaliza y escala (`primitives.py:114-204`).
- `sphere_vertex_count`, `sphere_triangle_count`, `icosphere_vertex_count`: helpers de conteo (`primitives.py:206-238`).
- `build_semicircle`: triángulo fan para semicírculo (centro + arco pi→0) (`primitives.py:240-289`).
- `build_cube`: vertices y normales por cara, índices de 12 triángulos (6 caras) (`primitives.py:291-355`).

### ar_filter/shaders

- `basic.vert`: recibe `position/normal`, calcula `fragPosition/fragNormal`, `gl_Position = projection*view*worldPos` (`shaders/basic.vert:1-20`).
- `basic.frag`: normaliza normal, iluminación difusa + ambiente, salida `vec4(result,1)` (`shaders/basic.frag:1-23`).
- `particle.vert`: usa `position/color/life`, fija `gl_PointSize=6*life`, pasa `fragColor/fragLife`, aplica view/projection (`shaders/particle.vert:1-24`).
- `particle.frag`: alpha = `fragLife*0.9` con `smoothstep` para borde suave en sprite de punto (`shaders/particle.frag:1-20`).

### tests/ (contratos de tests)

- `test_ar_filter_metrics.py`: valida métricas de `metrics.py` (clamp, widths, color functions, bar heights, yaw/roll, eyes).  
- `test_ar_filter_primitives.py`: comprueba conteo de vértices/triángulos de geometría.  
- `test_ar_filter_smoke.py`: prueba construcción de FaceTracker y OrbitingCubes (sin OpenGL).  
- `test_color_tracking.py`: asegura detección HSV y dibujo de ColorPainter.  
- `test_neon_menu.py` y `test_neon_menu_integration.py`: validan estados de animación, hit-test y callbacks del menú.  
- `test_scale_lock_ui.py`: verifica locking de escala y overlays en main/Gesture3D.  
- `test_spawn_position.py`: comprueba `_posicion_segura_creacion`.  
- `test_rotation.py`: revisa rotación de figuras.  
- `test_filters.py`, `test_geometry_utils.py`: cubren EMAFilter y rotate_points.

---

## 3) Secciones obligatorias

### A) MediaPipe FaceMesh / Landmarks

- **Inicialización FaceMesh:** `FaceTracker.__init__` crea `mp.solutions.face_mesh.FaceMesh` con `refine_landmarks=True`, `max_num_faces=1`, `min_detection_confidence=0.5`, `min_tracking_confidence=0.5` (`face_tracker.py:32-53`).
- **Landmarks usados:**
  - **Boca robot:** apertura usa índices 13 (labio superior) y 14 (inferior) para `mouth_openness`; centro combina 13/14/61/291; anchura usa 61 y 291 (`metrics.py:120-349`).
  - **Ojos:** apertura izquierda 159/145 y ancho 33/133; apertura derecha 386/374 y ancho 362/263; centros de ojo promedian esquinas y párpados (`metrics.py:648-748`).
  - **Anclaje cubos/medusas (halo):** usa `forehead_center` índice 10 y `face_width` con mejillas 234/454; halo radius derivado de ancho (`metrics.py:106-235`).
- **Normalización:** distancias se calculan en coordenadas normalizadas [0,1]; `mouth_openness` normaliza con MIN_DIST=0.01, MAX_DIST=0.06; `eye_openness` normaliza ratio párpado/anchura entre 0.05 y 0.25; `halo_radius` clampa 0.05–0.5 para tamaño consistente relativo al rostro.

### B) Módulo de métricas (ar_filter/metrics.py)

- Listado completo ya descrito arriba; cada función es pura matemática sin dependencias de OpenGL ni OpenCV. Devuelven floats/tuplas normalizadas para conducir tamaños/colores/rotaciones en `gl_app.py`. Se usan para:
  - Halo: `face_width`, `halo_radius`, `halo_sphere_positions_v2`.
  - Boca: `mouth_openness`, `mouth_center`, `mouth_width`, `mouth_plate_dimensions`, `robot_mouth_bar_color`, `robot_mouth_bar_heights`, `estimate_face_roll`.
  - Ojos: `left/right_eye_openness`, `left/right_eye_center`, `left/right_eye_width`, `eye_plate_dimensions`, `robot_eye_bar_color`, `robot_eye_bar_heights`.
  - Suavizado genérico: `smooth_value` para `smoothed_mouth` y `smoothed_left/right_eye` en `gl_app.py`.

### C) Render (OpenGL)

- **Shaders:** `basic.vert/basic.frag` para objetos (esferas, quads, cubos, semicírculos); `particle.vert/frag` para estelas de partículas; shaders embebidos simples para fondo y FaceMesh debug (`gl_app.py:405-482`, `shaders/*`).
- **VAOs/VBOs:** creados una vez en `_init_geometry`:
  - Esfera: `sphere_vao/vbo/nbo/ibo`, atributos `position` y `normal` (`gl_app.py:484-542`).
  - Quad: `quad_vao/vbo/nbo/ibo` comparte `pos_loc/norm_loc` (`gl_app.py:544-575`).
  - Fondo: `bg_vao/bg_vbo` con atributos `position` (2 floats) y `texCoord` (2 floats), stride 16 bytes; textura 2D para frame de cámara (`gl_app.py:577-639`).
  - FaceMesh debug: `mesh_vao/mesh_vbo` dinámico, `glBufferData` inicial 478*2*4 bytes; atributo `position` (vec2) (`gl_app.py:641-683`).
  - Cubo: `cube_vao/vbo/nbo/ibo` (`gl_app.py:685-731`).
  - Partículas: `particle_vao/vbo`, atributos position(3), color(3), life(1), stride 28 bytes (`gl_app.py:733-761`).
  - Semicírculo: `semicircle_vao/vbo/nbo/ibo` (`gl_app.py:763-804`).
- **Orden de draw calls:**
  1) Limpieza buffers. 2) Fondo con shader de fondo y textura de cámara (depth off). 3) FaceMesh debug (si `show_debug_mesh`: depth off). 4) Halo: partículas primero (`_render_particles`, blending on) y luego cubos (`_render_cube`). 5) Boca robótica (quads con blending). 6) Ojos robóticos (semicírculos + barras con blending). 7) Swap buffers (`gl_app.py:1409-1455`).
- **Blending/depth:** depth test habilitado globalmente; se desactiva para fondo y FaceMesh; blending activado en partículas, boca y ojos con `glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)` (`gl_app.py:1021-1052, 1188-1292, 1294-1397`).
- **Transformaciones:** matrices model/view/projection ortográficas; `_create_model_matrix` aplica escala uniforme y traslación; cubos aplican rotaciones Y/X/Z multiplicadas en orden `trans @ rot_y @ rot_x @ rot_z @ scale` (`gl_app.py:917-979`).

### D) Sistema de partículas/estelas

- **Estructura:** clase `Particle` con `pos[3]`, `vel[3]`, `life`, `color` (`gl_app.py:79-90`).
- **Pooling:** lista fija `particles` de longitud `PARTICLE_COUNT=500`; índice circular `particle_emit_idx` reutiliza slots (`gl_app.py:106-130`).
- **Emisión:** en `emit_trail_particle`, se usa posición del cubo y dirección tangencial inversa, velocidad aleatoria 0.02–0.05; se emiten 2 partículas por cubo por frame dentro de `update` (`gl_app.py:106-168`).
- **Update:** `update` suma `vel*dt` a `pos`, decrementa `life` con factor `dt*0.8`; si `life<=0` se ignora en render (`gl_app.py:148-168`).
- **Buffer por frame:** `get_particle_data` genera array contiguo [x,y,z,r,g,b,life] por partícula (0 si muerta) y se sube con `glBufferSubData` cada frame (`gl_app.py:170-206, 1021-1052`).

### E) NeonMenu / UI / Inputs / Estados

- **Estado del menú:** `MenuState` en `NeonMenu`; arranca `HIDDEN`, pasa por `OPENING`/`VISIBLE`/`CLOSING` según `open/close` (`neon_menu.py:24-115`).
- **Gestos de coordinación:** `PizarraNeon._actualizar_neon_menu` llama `Gesture3D.consume_menu_toggle` (gesto VICTORY) para abrir/cerrar; `cursor_pos` proviene de `Gesture3D.index_tip` o `last_pinch_position`; `is_selecting` se activa cuando pinch inicia (`main.py:838-866`).
- **Evitar interferencia:** cuando menú visible, `Gesture3D.set_rotation_enabled(False)` y `set_external_menu_active(True)` para bloquear rotaciones y lógica normal (`main.py:622-666`).
- **Modos de selección:** `SelectionMode.NORMAL` mueve/crea/elimina; `SelectionMode.SCALE` habilita escala por dedos o zonas. Transición con tecla `s`/`e` o toggle en gestos; `scale_lock_active` en `PizarraNeon` fuerza solo escala cuando pinch está en zonas laterales (`main.py:402-453, 710-774`).
- **Atajos de teclado:**
  - Global: `q` salir, `1` color, `2` gestos, `3` lanza AR (solo en menú), `m` vuelve a menú, `f` toggle overlay perf (`main.py:604-675`).
  - Color: `space` limpiar canvas, `c` color siguiente, `+/-` tamaño pincel, `1-6` presets HSV, `h` toggle calibración, `r` reset preset (`main.py:677-708`).
  - Gestos: `1-5` crean figuras, `x` clear, `space` delete seleccionada, `s` toggle scale, `e` salir de scale; con `scale_lock_active` solo `s/e` hacen efecto (`main.py:710-774`).
  - AR/GLFW: `ESC` cierra ventana, `D` toggle debug FaceMesh, `A/R/V/B` cambian `tracking_color` del mesh (`gl_app.py:333-383`).
- **Gatekeeping por frame:** `main.py` decide pipeline según `modo_actual`; `scale_lock_active` selecciona ruta reducida sin menú ni rotación; `Gesture3D.external_menu_active` deshabilita rotación continua (`main.py:593-666`).

### F) Contratos de tests

- Cubren métricas AR (`test_ar_filter_metrics.py`), geometría (`test_ar_filter_primitives.py`), rotaciones (`test_rotation.py`), spawn seguro (`test_spawn_position.py`), locking de escala UI (`test_scale_lock_ui.py`), color tracking (`test_color_tracking.py`), menú neon (`test_neon_menu*.py`), filtros EMA y utilidades geométricas (`test_filters.py`, `test_geometry_utils.py`), smoke de AR (`test_ar_filter_smoke.py`). Garantizan que cálculos de métricas, conteos de vértices, estados de menú y reglas de escala/rotación se mantengan según runtime actual.

---

## Lista de estados globales
- `PizarraNeon`: `modo_actual`, `cap`, `gesture_3d`, `color_painter`, `neon_menu`, `fps_actual`, caches de grid, `scale_lock_active`, `ultima_pos_cursor`, `debug_perf`, `perf_metrics` (`main.py`).
- `Gesture3D`: `figures`, `selected_figure`, `pinch_active`, `last_pinch_position`, `pinch_start_position`, `thumb_tip/index_tip`, `current_finger_distance`, `menu_state/menu_toggle_requested`, `selection_mode`, `_scale_zone_active`, `rotation_enabled`, `external_menu_active`, `pinch_filter` state (`Gesture3D.py`).
- `ColorPainter`: `canvas`, `brush_size`, `brush_color`, `current_color_index`, `last_pos`, `trail_particles`, `hsv_lower/upper`, `hsv_calibration_mode`, `last_mask`, `centroid_filter` state (`ColorPainter.py`).
- `ARFilterApp`: `running`, `window`, `cap`, `shader_programs`, VAOs/VBOs, `face_tracker`, `cubes_system.time/particles`, `halo_angle`, `smoothed_mouth`, `robot_mouth_time`, `robot_eye_time`, `smoothed_left_eye`, `smoothed_right_eye`, `tracking_color`, `show_debug_mesh`, FPS counters (`gl_app.py`).
- `OrbitingCubesSystem`: `cubes` params, `particles`, `particle_emit_idx`, `time` (`gl_app.py`).
- `NeonMenu`: `state`, `animation`, `_hover_index`, `_prev_selecting`, `_time_accum`, cache de posiciones (`neon_menu.py`).
- `FaceTracker`: `_last_landmarks` (cache de últimos landmarks válidos) (`face_tracker.py`).

## Lista de parámetros sensibles
- Thresholds HSV (`ColorPainter.HSV_PRESETS`, `hsv_lower/upper`), área mínima de contorno 300 px², cache máscara 0.1s (`ColorPainter.py`).
- EMA alpha=0.4 para centroides y pinch (`ColorPainter.py`, `Gesture3D.py`).
- Límites de escala de figuras: tamaño 15–300 px, `min/max_finger_distance 20/250`, `_spatial_scale_speed 150`, `size_smoothing 0.2` (`Gesture3D.py`).
- Halo/barras AR: `halo_radius` clamp 0.05–0.5, `mouth_openness` rango 0.01–0.06, `eye_openness` rango 0.05–0.25, `NUM_HALO_SPHERES=8`, `CUBE_SIZE=0.017`, partículas 500, `robot_mouth` bars=7 con `pulse_freq=10`, `robot_eye` pulse_freq=15 (`metrics.py`, `gl_app.py`).
- Animaciones menú: `open_duration 0.35s`, `close_duration 0.25s`, `hover_fade 0.25s`, `pulse_speed 2.2` (`neon_menu.py`).
- HUD principal: `grid_update_interval=0.2`, `scale_zone_left/right_pct=0.33/0.67`, márgenes verticales 12% (`main.py`).

---

## Confirmación final

- **Lista de archivos analizados:** main.py, Gesture3D.py, ColorPainter.py, neon_menu.py, neon_effects.py, geometry_utils.py, filters.py, ar_filter/gl_app.py, ar_filter/metrics.py, ar_filter/primitives.py, ar_filter/face_tracker.py, ar_filter/shaders/*.vert/*.frag, tests/*.
- **Funciones/clases descritas (aprox):** ~120 entre métodos, helpers, clases y enums.
- **Checklist:** cubrió filtros ✅, menú ✅, inputs ✅, mediapipe ✅, render ✅, partículas ✅, tests ✅.
