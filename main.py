import math
import cv2
import numpy as np
import time
import os
from collections import deque
from ColorPainter import ColorPainter
from Gesture3D import Gesture3D, SelectionMode, Gesture
from neon_menu import MenuButton, NeonMenu

# Configuracion para Ubuntu/Wayland
os.environ['QT_QPA_PLATFORM'] = 'xcb'


class FPSCounter:
    """Moving average FPS counter for real performance metrics."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_time = time.perf_counter()

    def tick(self) -> float:
        """Call once per frame. Returns current FPS."""
        now = time.perf_counter()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0:
            self.frame_times.append(dt)
        return self.get_fps()

    def get_fps(self) -> float:
        """Get moving average FPS."""
        if not self.frame_times:
            return 0.0
        avg_dt = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_dt if avg_dt > 0 else 0.0

    def get_dt(self) -> float:
        """Get last frame delta time."""
        if self.frame_times:
            return self.frame_times[-1]
        return 0.033  # Default ~30 FPS


class PizarraNeon:
    def __init__(self):
        self.cap = None
        self.modo_actual = "menu"
        self.ancho = 1280
        self.alto = 720
        self.tiempo_inicio = time.time()

        # Inicializar modulos
        self.color_painter = ColorPainter(self.ancho, self.alto)
        self.gesture_3d = Gesture3D(self.ancho, self.alto, use_external_menu=True)
        self.neon_menu = self._crear_neon_menu()
        self.prev_pinch_activo = False
        self.ultima_pos_cursor = (self.ancho // 2, self.alto // 2)

        # Performance tracking
        self.fps_counter = FPSCounter(window_size=30)
        self.debug_perf = False
        self.perf_metrics = {
            "total": 0.0,
            "detect": 0.0,
            "handle": 0.0,
            "draw_figures": 0.0,
            "menu_update": 0.0,
            "menu_draw": 0.0,
        }

        # Auto-disable fancy effects if FPS drops
        self.low_fps_threshold = 15
        self.low_fps_mode = False

        # Cache para optimizacion (grid solo)
        self.grid_cache = None
        self.ultimo_grid_update = 0
        self.grid_update_interval = 0.3

        # Estado de bloqueo de escala
        self.scale_lock_active = False

        # Paleta de colores
        self.colores = {
            'azul_electrico': (255, 120, 0),
            'azul_claro': (255, 200, 100),
            'azul_oscuro': (180, 80, 20),
            'cyan_tech': (255, 255, 80),
            'verde_matrix': (80, 255, 0),
            'blanco_tech': (220, 220, 200),
            'fondo': (5, 5, 10),
            'panel': (12, 12, 18),
            'gris_tech': (100, 100, 120),
            'verde_seleccion': (0, 180, 0)
        }

    def _crear_neon_menu(self):
        """Configura el menu neon con callbacks de figuras."""

        def crear_callback(figura):
            def _callback(_):
                # Create figure at safe position
                posicion = self._posicion_segura_creacion()
                self.gesture_3d.create_figure(figura, posicion)
                # Force minimum visible size
                if self.gesture_3d.selected_figure:
                    self.gesture_3d.selected_figure['size'] = max(
                        40, self.gesture_3d.selected_figure['size']
                    )
                # Close menu IMMEDIATELY in same frame
                self.neon_menu.close()
            return _callback

        def eliminar_callback(_):
            self.gesture_3d.delete_selected_figure()
            self.neon_menu.close()

        palette = [
            (255, 140, 80),
            (120, 255, 200),
            (255, 120, 200),
            (90, 200, 255),
            (255, 210, 120),
            (160, 120, 255),
        ]

        botones = []
        for idx, figura in enumerate(self.gesture_3d.available_figures):
            color = palette[idx % len(palette)]
            botones.append(MenuButton(figura, color, on_select=crear_callback(figura)))

        botones.append(MenuButton("delete", (60, 60, 255), on_select=eliminar_callback))

        return NeonMenu(
            center=(self.ancho // 2, self.alto // 2),
            radius=85,
            button_radius=20,
            inner_deadzone=26,
            buttons=botones,
        )

    def inicializar(self):
        """Inicializa todos los modulos del sistema"""
        print("=" * 45)
        print("  INICIANDO SISTEMA NEURAL v2.1 (STABLE)")
        print("  > Inicializando modulos...")
        print("=" * 45)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.ancho)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.alto)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            print("[ ERROR ] No se pudo acceder al dispositivo de captura")
            return False

        print("[ OK ] Modulo de captura: ONLINE")
        print("[ OK ] Procesador visual: ACTIVO")
        print("[ OK ] ColorPainter: INICIALIZADO")
        print("[ OK ] Gesture3D: INICIALIZADO")
        return True

    def _posicion_segura_creacion(self):
        """Ajusta la posicion de spawn para garantizar 100% visibilidad."""
        margen = 80
        min_size = 40

        # Prefer pinch position if valid and inside frame
        preferida = None
        if self.gesture_3d.last_pinch_position:
            px, py = self.gesture_3d.last_pinch_position
            if margen <= px <= self.ancho - margen and margen <= py <= self.alto - margen:
                preferida = (px, py)

        if preferida is None:
            preferida = self.ultima_pos_cursor

        if preferida is None:
            preferida = (self.ancho // 2, self.alto // 2)

        # Clamp to safe bounds (accounting for figure size)
        safe_margin = margen + min_size
        x = min(self.ancho - safe_margin, max(safe_margin, int(preferida[0])))
        y = min(self.alto - safe_margin, max(safe_margin, int(preferida[1])))
        return (x, y)

    @staticmethod
    def dibujar_texto_limpio(image, text, position, font_scale, color, thickness=1):
        """Texto limpio SIN glow - estilo terminal"""
        shadow_color = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
        cv2.putText(image, text, (position[0] + 1, position[1] + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, shadow_color, thickness, cv2.LINE_AA)
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)

    def dibujar_grid_minimal(self, frame, spacing=80):
        """Grid MINIMALISTA con cache - sin pulse"""
        current_time = time.time() - self.tiempo_inicio

        if self.grid_cache is not None and (current_time - self.ultimo_grid_update) < self.grid_update_interval:
            frame[:] = cv2.addWeighted(frame, 1.0, self.grid_cache, 0.10, 0)
            return

        self.grid_cache = np.zeros_like(frame)
        grid_color = (25, 18, 15)

        for x in range(0, self.ancho, spacing * 2):
            cv2.line(self.grid_cache, (x, 0), (x, self.alto), grid_color, 1, cv2.LINE_AA)

        for y in range(0, self.alto, spacing * 2):
            cv2.line(self.grid_cache, (0, y), (self.ancho, y), grid_color, 1, cv2.LINE_AA)

        self.ultimo_grid_update = current_time
        frame[:] = cv2.addWeighted(frame, 1.0, self.grid_cache, 0.10, 0)

    def dibujar_borde_esquinas(self, frame):
        """Borde solo en las esquinas - estilo HUD (sin pulse)"""
        color = self.colores['azul_electrico']
        margin = 15
        length = 40
        thickness = 2

        corners = [
            (margin, margin), (self.ancho - margin, margin),
            (margin, self.alto - margin), (self.ancho - margin, self.alto - margin)
        ]

        for i, (x, y) in enumerate(corners):
            if i == 0:  # Top-left
                cv2.line(frame, (x, y), (x + length, y), color, thickness, cv2.LINE_AA)
                cv2.line(frame, (x, y), (x, y + length), color, thickness, cv2.LINE_AA)
            elif i == 1:  # Top-right
                cv2.line(frame, (x - length, y), (x, y), color, thickness, cv2.LINE_AA)
                cv2.line(frame, (x, y), (x, y + length), color, thickness, cv2.LINE_AA)
            elif i == 2:  # Bottom-left
                cv2.line(frame, (x, y - length), (x, y), color, thickness, cv2.LINE_AA)
                cv2.line(frame, (x, y), (x + length, y), color, thickness, cv2.LINE_AA)
            elif i == 3:  # Bottom-right
                cv2.line(frame, (x - length, y), (x, y), color, thickness, cv2.LINE_AA)
                cv2.line(frame, (x, y - length), (x, y), color, thickness, cv2.LINE_AA)

    def _get_scale_zone_rects(self):
        """Devuelve los rectángulos de zona de escala (left, right)."""
        x_left = 0
        x_center_left = int(self.ancho * (1 / 3))
        x_center_right = int(self.ancho * (2 / 3))
        y_top = int(self.alto * 0.15)
        y_bottom = int(self.alto * 0.85)
        left_rect = (x_left, y_top, x_center_left, y_bottom)
        right_rect = (x_center_right, y_top, self.ancho, y_bottom)
        return left_rect, right_rect

    def _draw_scale_lock_overlay(self, frame, left_rect, right_rect, alpha=0.3):
        """Dibuja las zonas de escala cuando el modo está activo."""
        overlay = frame.copy()
        fill_color = (200, 80, 30)
        border_color = (255, 160, 90)

        def _draw_zone(rect, label):
            x1, y1, x2, y2 = rect
            overlay[y1:y2, x1:x2] = fill_color
            cv2.rectangle(overlay, (x1, y1), (x2, y2), border_color, 2)
            cv2.putText(overlay, label, (x1 + 20, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, border_color, 2, cv2.LINE_AA)

        _draw_zone(left_rect, "- SIZE")
        _draw_zone(right_rect, "+ SIZE")
        blended = (overlay.astype(np.float32) * alpha + frame.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        frame[:] = blended
        cv2.putText(frame, "MODE: SCALE (PINCH in blue zones)", (int(self.ancho * 0.25), 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

    def dibujar_display_seleccion(self, frame):
        """Display verde que muestra la figura seleccionada"""
        if not self.gesture_3d.selected_figure:
            return

        figura = self.gesture_3d.selected_figure
        header_height = 70

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.ancho, header_height), (0, 60, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        cv2.rectangle(frame, (0, 0), (self.ancho, header_height), self.colores['verde_matrix'], 2, cv2.LINE_AA)

        tipo_figura = figura['type'].upper()
        tamano = figura['size']
        texto_principal = f"FIGURA SELECCIONADA: {tipo_figura} - Tamano: {tamano}px"

        self.dibujar_texto_limpio(frame, texto_principal, (20, 35),
                                  0.8, self.colores['verde_matrix'], 2)

        cv2.rectangle(frame, (self.ancho - 150, 15), (self.ancho - 20, 55),
                      self.colores['verde_matrix'], 2, cv2.LINE_AA)

        self._dibujar_miniatura_figura(frame, (self.ancho - 85, 35), figura['type'])

    def _dibujar_miniatura_figura(self, frame, position, fig_type):
        """Dibuja una miniatura de la figura seleccionada"""
        x, y = position
        color = self.colores['verde_matrix']
        tamano = 12

        if fig_type == 'circle':
            cv2.circle(frame, (x, y), tamano, color, 2, cv2.LINE_AA)
        elif fig_type == 'square':
            cv2.rectangle(frame, (x - tamano, y - tamano), (x + tamano, y + tamano), color, 2, cv2.LINE_AA)
        elif fig_type == 'triangle':
            pts = np.array([[x, y - tamano], [x - tamano, y + tamano], [x + tamano, y + tamano]], np.int32)
            cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
        elif fig_type == 'star':
            self._dibujar_estrella_mini(frame, (x, y), tamano, color)
        elif fig_type == 'heart':
            self._dibujar_corazon_mini(frame, (x, y), tamano, color)

    def _dibujar_estrella_mini(self, frame, center, size, color):
        """Dibuja una estrella miniatura"""
        x, y = center
        points = []
        for i in range(10):
            angle = math.pi / 2 + i * math.pi / 5
            r = size if i % 2 == 0 else size / 2
            points.append((int(x + r * math.cos(angle)), int(y + r * math.sin(angle))))

        pts = np.array(points, np.int32)
        cv2.polylines(frame, [pts], True, color, 1, cv2.LINE_AA)

    def _dibujar_corazon_mini(self, frame, center, size, color):
        """Dibuja un corazon miniatura"""
        x, y = center
        cv2.ellipse(frame, (x - size // 2, y - size // 3), (size // 2, size // 3), 0, 0, 180, color, 1, cv2.LINE_AA)
        cv2.ellipse(frame, (x + size // 2, y - size // 3), (size // 2, size // 3), 0, 0, 180, color, 1, cv2.LINE_AA)
        pts = np.array([[x, y + size // 2], [x - size, y - size // 3], [x + size, y - size // 3]], np.int32)
        cv2.polylines(frame, [pts], True, color, 1, cv2.LINE_AA)

    def dibujar_menu_principal(self, frame):
        """Menu principal optimizado"""
        frame[:] = self.colores['fondo']
        self.dibujar_grid_minimal(frame, spacing=100)
        self.dibujar_borde_esquinas(frame)

        titulo = "[ NEURAL CANVAS v2.1 ]"
        x_titulo = self.ancho // 2 - 200
        self.dibujar_texto_limpio(frame, titulo, (x_titulo, 100),
                                  1.2, self.colores['azul_electrico'], 2)

        cv2.line(frame, (x_titulo - 20, 120), (x_titulo + 440, 120),
                 self.colores['azul_oscuro'], 1, cv2.LINE_AA)

        menu_items = [
            {"prefijo": "[1]", "texto": "TRACKING MODULE", "desc": "Color Detection & Painting System",
             "color": self.colores['azul_electrico']},
            {"prefijo": "[2]", "texto": "GESTURE MODULE", "desc": "3D Hand Recognition & Figure Control",
             "color": self.colores['cyan_tech']},
            {"prefijo": "[Q]", "texto": "EXIT", "desc": "Shutdown System",
             "color": self.colores['gris_tech']}
        ]

        y_start = 220
        for i, item in enumerate(menu_items):
            y_pos = y_start + i * 90

            self.dibujar_texto_limpio(frame, item["prefijo"],
                                      (self.ancho // 2 - 250, y_pos),
                                      0.8, item["color"], 2)

            self.dibujar_texto_limpio(frame, item["texto"],
                                      (self.ancho // 2 - 180, y_pos),
                                      0.9, self.colores['blanco_tech'], 2)

            self.dibujar_texto_limpio(frame, f"> {item['desc']}",
                                      (self.ancho // 2 - 180, y_pos + 25),
                                      0.5, self.colores['gris_tech'], 1)

            if i < len(menu_items) - 1:
                y_line = y_pos + 50
                cv2.line(frame, (self.ancho // 2 - 250, y_line),
                         (self.ancho // 2 + 250, y_line),
                         self.colores['azul_oscuro'], 1, cv2.LINE_AA)

        footer_y = self.alto - 40
        self.dibujar_texto_limpio(frame, "COMPUTER VISION LAB | 2025",
                                  (self.ancho // 2 - 180, footer_y),
                                  0.6, self.colores['gris_tech'], 1)

        return frame

    def dibujar_hud_superior(self, frame, texto_modo):
        """HUD superior con FPS real"""
        cv2.rectangle(frame, (0, 0), (self.ancho, 45), self.colores['panel'], -1)

        self.dibujar_texto_limpio(frame, f">> {texto_modo}", (20, 30),
                                  0.6, self.colores['azul_electrico'], 1)

        fps = self.fps_counter.get_fps()
        fps_text = f"FPS: {fps:.1f}"
        color_fps = self.colores['verde_matrix'] if fps > 25 else (
            self.colores['azul_electrico'] if fps > 15 else (0, 100, 255)
        )
        self.dibujar_texto_limpio(frame, fps_text, (self.ancho - 120, 30),
                                  0.5, color_fps, 1)

        # Low FPS warning
        if self.low_fps_mode:
            self.dibujar_texto_limpio(frame, "[LOW FPS MODE]", (self.ancho - 280, 30),
                                      0.4, (0, 100, 255), 1)

    def _dibujar_overlay_perf(self, frame):
        """Panel ligero con metricas de performance (toggle con tecla F)."""
        panel_x, panel_y = 15, 60
        ancho, alto = 220, 140
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + ancho, panel_y + alto), (15, 15, 20), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + ancho, panel_y + alto), (60, 100, 180), 1)

        fps = self.fps_counter.get_fps()
        lineas = [
            f"FPS: {fps:.1f}",
            f"total : {self.perf_metrics['total']*1000:.1f} ms",
            f"detect: {self.perf_metrics['detect']*1000:.1f} ms",
            f"logic : {self.perf_metrics['handle']*1000:.1f} ms",
            f"figures: {self.perf_metrics['draw_figures']*1000:.1f} ms",
            f"menu up: {self.perf_metrics['menu_update']*1000:.1f} ms",
            f"menu dr: {self.perf_metrics['menu_draw']*1000:.1f} ms",
        ]

        for i, texto in enumerate(lineas):
            self.dibujar_texto_limpio(frame, texto, (panel_x + 10, panel_y + 20 + i * 18), 0.42,
                                      self.colores['azul_electrico'], 1)

    def modo_seguimiento_color(self, frame):
        """Modo tracking optimizado"""
        self.dibujar_grid_minimal(frame)
        self.dibujar_borde_esquinas(frame)
        self.dibujar_hud_superior(frame, "COLOR TRACKING MODULE")

        frame_procesado = self.color_painter.process_frame(frame)
        return frame_procesado

    def modo_figuras_gestos(self, frame):
        """Modo gestos con ORDEN DE DIBUJO ESTRICTO:
        1. Frame base (grid, bordes, HUD)
        2. Figuras (via Gesture3D)
        3. NeonMenu (si visible)
        4. Overlay debug (si activo)
        """
        dt = self.fps_counter.get_dt()

        # Check for low FPS mode
        fps = self.fps_counter.get_fps()
        if fps > 0 and fps < self.low_fps_threshold:
            self.low_fps_mode = True
        elif fps > self.low_fps_threshold + 5:
            self.low_fps_mode = False

        # === STEP 1: Frame base ===
        self.dibujar_grid_minimal(frame)
        self.dibujar_borde_esquinas(frame)
        self.dibujar_hud_superior(frame, "GESTURE RECOGNITION MODULE")
        self.dibujar_display_seleccion(frame)

        # === STEP 2: Gesture detection and figure drawing ===
        profile = {} if self.debug_perf else None

        detect_start = time.perf_counter()
        gesture, raw_pinch, hand_landmarks = self.gesture_3d.detect_gestures(frame)
        detect_dt = time.perf_counter() - detect_start

        # Update pinch position
        pinch_position = self.gesture_3d.pinch_filter.update(raw_pinch)
        if pinch_position:
            self.gesture_3d.last_pinch_position = pinch_position

        self.scale_lock_active = (
            self.gesture_3d.selection_mode == SelectionMode.SCALE
            and self.gesture_3d.selected_figure is not None
        )

        if self.scale_lock_active and self.neon_menu.is_visible():
            self.neon_menu.close()

        menu_activo = self.neon_menu.is_visible()
        self.gesture_3d.set_rotation_enabled(not (menu_activo or self.scale_lock_active))
        self.gesture_3d.set_external_menu_active(menu_activo)

        left_rect = right_rect = None
        handle_start = time.perf_counter()
        if self.scale_lock_active:
            left_rect, right_rect = self._get_scale_zone_rects()
            self._manejar_escala_bloqueada(gesture, pinch_position, dt, left_rect, right_rect)
        else:
            self.gesture_3d.handle_gestures(gesture, pinch_position, time.time())
        handle_dt = time.perf_counter() - handle_start

        # Draw figures (Gesture3D internal UI: figures, landmarks, pinch cursor)
        draw_start = time.perf_counter()
        self.gesture_3d.draw_interface(frame, gesture, pinch_position, hand_landmarks)
        if self.scale_lock_active and left_rect and right_rect:
            self._draw_scale_lock_overlay(frame, left_rect, right_rect)
        draw_dt = time.perf_counter() - draw_start

        # === STEP 3: Update and draw NeonMenu ===
        menu_update_dt = 0.0
        menu_draw_dt = 0.0
        if not self.scale_lock_active:
            update_start = time.perf_counter()
            self._actualizar_neon_menu(dt)
            menu_update_dt = time.perf_counter() - update_start

            menu_draw_start = time.perf_counter()
            # ONLY draw menu here - nowhere else
            if self.neon_menu.is_visible():
                self.neon_menu.draw(frame)
            menu_draw_dt = time.perf_counter() - menu_draw_start

        # === STEP 4: Debug overlay (if enabled) ===
        if self.debug_perf:
            self.perf_metrics = {
                "total": detect_dt + handle_dt + draw_dt + menu_update_dt + menu_draw_dt,
                "detect": detect_dt,
                "handle": handle_dt,
                "draw_figures": draw_dt,
                "menu_update": menu_update_dt,
                "menu_draw": menu_draw_dt,
            }
            self._dibujar_overlay_perf(frame)

        return frame

    def procesar_teclas(self, key):
        """Procesa todas las entradas de teclado de manera centralizada"""
        if key == ord('q'):
            print("\n[ SHUTDOWN ] Cerrando sistema...")
            return 'quit'
        elif key == ord('1'):
            self.modo_actual = "color"
            print("[ MODE ] Color Tracking: ACTIVATED")
        elif key == ord('2'):
            self.modo_actual = "gestos"
            print("[ MODE ] Gesture Recognition: ACTIVATED")
        elif key == ord('m'):
            self.modo_actual = "menu"
            print("[ MODE ] Main Menu: LOADED")
        elif key == ord('f'):
            self.debug_perf = not self.debug_perf
            estado = "ON" if self.debug_perf else "OFF"
            print(f"[ PERF ] Overlay {estado}")

        elif self.modo_actual == "color":
            self._procesar_teclas_color(key)
        elif self.modo_actual == "gestos":
            self._procesar_teclas_gestos(key)

        return 'continue'

    def _procesar_teclas_color(self, key):
        """Procesa teclas especificas del modo color"""
        if key == ord(' '):
            self.color_painter.clear_canvas()
            print("[ ACTION ] Canvas cleared")
        elif key == ord('c'):
            new_color = self.color_painter.change_brush_color()
            print(f"[ ACTION ] Brush color changed to RGB{new_color}")
        elif key == ord('+'):
            new_size = self.color_painter.change_brush_size(True)
            print(f"[ ACTION ] Brush size increased to {new_size}")
        elif key == ord('-'):
            new_size = self.color_painter.change_brush_size(False)
            print(f"[ ACTION ] Brush size decreased to {new_size}")
        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')]:
            preset_num = key - ord('0')
            self.color_painter.set_hsv_preset(preset_num)
        elif key == ord('h'):
            is_active = self.color_painter.toggle_hsv_calibration()
            state = "ACTIVADA" if is_active else "DESACTIVADA"
            print(f"[ HSV ] Calibracion {state}")
        elif key == ord('r'):
            self.color_painter.reset_to_default_preset()
            print("[ HSV ] Reset a preset por defecto (AZUL)")

    def _procesar_teclas_gestos(self, key):
        """Procesa teclas especificas del modo gestos"""
        posicion_central = (self.ancho // 2, self.alto // 2)

        if self.scale_lock_active and key not in (ord('e'), ord('E'), ord('s')):
            # Solo permitir salida de modo escala o toggle explicito
            return

        if key in (ord('e'), ord('E')) and self.gesture_3d.selection_mode == SelectionMode.SCALE:
            self.gesture_3d.toggle_scale_mode()
            self.scale_lock_active = False
            return

        if key == ord('1'):
            self.gesture_3d.create_figure_by_key('circle', posicion_central)
            print("[ ACTION ] Circle created")
        elif key == ord('2'):
            self.gesture_3d.create_figure_by_key('square', posicion_central)
            print("[ ACTION ] Square created")
        elif key == ord('3'):
            self.gesture_3d.create_figure_by_key('triangle', posicion_central)
            print("[ ACTION ] Triangle created")
        elif key == ord('4'):
            self.gesture_3d.create_figure_by_key('star', posicion_central)
            print("[ ACTION ] Star created")
        elif key == ord('5'):
            self.gesture_3d.create_figure_by_key('heart', posicion_central)
            print("[ ACTION ] Heart created")
        elif key == ord('x'):
            self.gesture_3d.clear_figures()
            print("[ ACTION ] All figures cleared")
        elif key == ord(' '):
            self.gesture_3d.delete_selected_figure()
            print("[ ACTION ] Selected figure deleted")
        elif key == ord('s'):
            self.gesture_3d.toggle_scale_mode()

    def _actualizar_neon_menu(self, dt):
        """Sincroniza gestos con el menu neon."""
        self.ultima_pos_cursor = (
            self.gesture_3d.index_tip
            or self.gesture_3d.last_pinch_position
            or self.ultima_pos_cursor
        )

        pinch_activo = self.gesture_3d.pinch_active
        pinch_inicio = pinch_activo and not self.prev_pinch_activo
        self.prev_pinch_activo = pinch_activo

        toggled, gesture_center = self.gesture_3d.consume_menu_toggle()
        if toggled:
            if self.neon_menu.is_visible():
                self.neon_menu.close()
            else:
                self.neon_menu.center = gesture_center or self.ultima_pos_cursor
                self.neon_menu.open()

        self.neon_menu.update(self.ultima_pos_cursor, pinch_inicio, dt)

        # Update rotation enabled state after menu state changes
        menu_activo = self.neon_menu.is_visible()
        self.gesture_3d.set_rotation_enabled(not menu_activo)
        self.gesture_3d.set_external_menu_active(menu_activo)

    def _manejar_escala_bloqueada(self, gesture, pinch_position, dt, left_rect, right_rect):
        """Gestiona la interacción cuando el modo escala está bloqueado."""

        if gesture == Gesture.PINCH:
            if not self.gesture_3d.pinch_active:
                self.gesture_3d.pinch_active = True
            if pinch_position:
                self.gesture_3d.last_pinch_position = pinch_position
                self.gesture_3d.handle_figure_scaling_by_spatial(pinch_position, dt, left_rect, right_rect)
        else:
            if self.gesture_3d.pinch_active:
                self.gesture_3d.reset_spatial_scale_state()
                self.gesture_3d.reset_angular_scale_state()
            self.gesture_3d.pinch_active = False
            self.gesture_3d.pinch_start_position = None

    def ejecutar(self):
        """Bucle principal optimizado"""
        if not self.inicializar():
            return

        print("\n" + "=" * 45)
        print("  SISTEMA INICIADO CORRECTAMENTE")
        print("  [1] Color Tracking")
        print("  [2] Gesture Recognition")
        print("  [M] Main Menu")
        print("  [F] Toggle Performance Overlay")
        print("  [Q] Shutdown")
        print("=" * 45 + "\n")

        cv2.namedWindow('NEURAL CANVAS v2.1', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('NEURAL CANVAS v2.1', 1200, 700)

        try:
            while True:
                # Update FPS counter
                self.fps_counter.tick()

                ret, frame = self.cap.read()
                if not ret:
                    print("[ ERROR ] Frame capture failed")
                    break

                frame = cv2.flip(frame, 1)

                if frame.shape[1] != self.ancho or frame.shape[0] != self.alto:
                    frame = cv2.resize(frame, (self.ancho, self.alto))

                frame_procesado = frame.copy()
                if self.modo_actual == "menu":
                    frame_procesado = self.dibujar_menu_principal(frame_procesado)
                elif self.modo_actual == "color":
                    frame_procesado = self.modo_seguimiento_color(frame_procesado)
                elif self.modo_actual == "gestos":
                    frame_procesado = self.modo_figuras_gestos(frame_procesado)

                cv2.imshow('NEURAL CANVAS v2.1', frame_procesado)

                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    result = self.procesar_teclas(key)
                    if result == 'quit':
                        break

        except KeyboardInterrupt:
            print("\n[ INTERRUPT ] Sistema interrumpido por usuario")
        except Exception as e:
            print(f"[ CRITICAL ERROR ] {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.liberar_recursos()

    def liberar_recursos(self):
        """Libera todos los recursos del sistema"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("\n" + "=" * 45)
        print("  SISTEMA APAGADO CORRECTAMENTE")
        print("=" * 45)


if __name__ == "__main__":
    app = PizarraNeon()
    app.ejecutar()
