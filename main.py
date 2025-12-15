import math
import cv2
import numpy as np
import time
import os
from ColorPainter import ColorPainter
from Gesture3D import Gesture3D, SelectionMode, Gesture
from neon_menu import MenuButton, NeonMenu
from ar_filter import run_ar_filter

# Configuración para Ubuntu/Wayland
os.environ['QT_QPA_PLATFORM'] = 'xcb'


class PizarraNeon:
    def __init__(self):
        self.cap = None
        self.modo_actual = "menu"
        self.ancho = 1280
        self.alto = 720
        self.tiempo_inicio = time.time()

        # Inicializar módulos
        self.color_painter = ColorPainter(self.ancho, self.alto)
        self.gesture_3d = Gesture3D(self.ancho, self.alto, use_external_menu=True)
        self.neon_menu = self._crear_neon_menu()
        self.ultimo_dt = time.perf_counter()
        self.prev_pinch_activo = False
        self.ultima_pos_cursor = (self.ancho // 2, self.alto // 2)
        self.debug_perf = False
        self.perf_metrics = {
            "total": 0.0,
            "detect": 0.0,
            "handle": 0.0,
            "draw_figures": 0.0,
            "menu_update": 0.0,
            "menu_draw": 0.0,
        }

        # Cache para optimización
        self.grid_cache = None
        self.ultimo_grid_update = 0
        self.grid_update_interval = 0.2  # Increased for better performance

        # FPS counter
        self.fps_counter = 0
        self.fps_time = time.time()
        self.fps_actual = 0

        # Paleta de colores optimizada
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
        """Configura el menú neon con callbacks de figuras."""

        def crear_callback(figura):
            def _callback(_):
                posicion = self._posicion_segura_creacion()
                self.gesture_3d.create_figure(figura, posicion)
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
        """Inicializa todos los módulos del sistema"""
        print("╔═══════════════════════════════════════╗")
        print("║  INICIANDO SISTEMA NEURAL v2.1      ║")
        print("║  > Inicializando módulos...          ║")
        print("╚═══════════════════════════════════════╝")

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.ancho)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.alto)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            print("[ ERROR ] No se pudo acceder al dispositivo de captura")
            return False

        print("[ OK ] Módulo de captura: ONLINE")
        print("[ OK ] Procesador visual: ACTIVO")
        print("[ OK ] ColorPainter: INICIALIZADO")
        print("[ OK ] Gesture3D: INICIALIZADO")
        return True

    def _restaurar_captura(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.ancho)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.alto)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cv2.namedWindow('NEURAL CANVAS v2.1', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('NEURAL CANVAS v2.1', 1200, 700)

    def _posicion_segura_creacion(self):
        """Ajusta la posición de spawn para garantizar visibilidad."""

        margen = 80
        preferida = (
            self.gesture_3d.last_pinch_position
            or self.ultima_pos_cursor
            or (self.ancho // 2, self.alto // 2)
        )

        x = min(self.ancho - margen, max(margen, int(preferida[0])))
        y = min(self.alto - margen, max(margen, int(preferida[1])))
        return (x, y)

    def lanzar_filtro_ar(self):
        print("[ MODE ] AR Filter: LAUNCHING")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        try:
            run_ar_filter()
        finally:
            self._restaurar_captura()

    @staticmethod
    def dibujar_texto_limpio(image, text, position, font_scale, color, thickness=1):
        """Texto limpio SIN glow excesivo - estilo terminal"""
        shadow_color = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
        cv2.putText(image, text, (position[0] + 1, position[1] + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, shadow_color, thickness, cv2.LINE_AA)
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)

    def dibujar_grid_minimal(self, frame, spacing=80):
        """Grid MINIMALISTA con cache mejorado - OPTIMIZADO"""
        current_time = time.time() - self.tiempo_inicio

        # Cache más eficiente
        if self.grid_cache is not None and (current_time - self.ultimo_grid_update) < self.grid_update_interval:
            frame[:] = cv2.addWeighted(frame, 1.0, self.grid_cache, 0.12, 0)
            return

        # Regenerar grid solo cuando es necesario
        self.grid_cache = np.zeros_like(frame)
        pulse = (math.sin(current_time * 0.8) + 1) / 2  # Slower pulse for performance

        base = 15 + int(pulse * 8)  # Reduced intensity
        grid_color = (base + 10, base + 3, base)

        # Draw fewer lines for better performance
        for x in range(0, self.ancho, spacing * 2):  # Only every other line
            cv2.line(self.grid_cache, (x, 0), (x, self.alto), grid_color, 1, cv2.LINE_AA)

        for y in range(0, self.alto, spacing * 2):
            cv2.line(self.grid_cache, (0, y), (self.ancho, y), grid_color, 1, cv2.LINE_AA)

        self.ultimo_grid_update = current_time
        frame[:] = cv2.addWeighted(frame, 1.0, self.grid_cache, 0.12, 0)

    def dibujar_borde_esquinas(self, frame):
        """Borde solo en las esquinas - estilo HUD optimizado"""
        current_time = time.time() - self.tiempo_inicio
        pulse = (math.sin(current_time * 1.5) + 1) / 2  # Slower animation

        color = (
            int(self.colores['azul_electrico'][0] * (0.7 + pulse * 0.3)),
            int(self.colores['azul_electrico'][1] * (0.7 + pulse * 0.3)),
            int(self.colores['azul_electrico'][2] * (0.7 + pulse * 0.3))
        )

        margin = 15
        length = 40
        thickness = 2

        # Esquinas simplificadas
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

    def dibujar_display_seleccion(self, frame):
        """Display verde que muestra la figura seleccionada"""
        if not self.gesture_3d.selected_figure:
            return

        figura = self.gesture_3d.selected_figure
        header_height = 70

        # Fondo del display
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.ancho, header_height), (0, 60, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Borde superior
        cv2.rectangle(frame, (0, 0), (self.ancho, header_height), self.colores['verde_matrix'], 2, cv2.LINE_AA)

        # Texto de información
        tipo_figura = figura['type'].upper()
        tamaño = figura['size']
        texto_principal = f"FIGURA SELECCIONADA: {tipo_figura} - Tamaño: {tamaño}px"

        self.dibujar_texto_limpio(frame, texto_principal, (20, 35),
                                  0.8, self.colores['verde_matrix'], 2)

        # Indicador visual a la derecha
        cv2.rectangle(frame, (self.ancho - 150, 15), (self.ancho - 20, 55),
                      self.colores['verde_matrix'], 2, cv2.LINE_AA)

        # Miniatura de la figura
        self._dibujar_miniatura_figura(frame, (self.ancho - 85, 35), figura['type'])

    def _dibujar_miniatura_figura(self, frame, position, fig_type):
        """Dibuja una miniatura de la figura seleccionada"""
        x, y = position
        color = self.colores['verde_matrix']
        tamaño = 12

        if fig_type == 'circle':
            cv2.circle(frame, (x, y), tamaño, color, 2, cv2.LINE_AA)
        elif fig_type == 'square':
            cv2.rectangle(frame, (x - tamaño, y - tamaño), (x + tamaño, y + tamaño), color, 2, cv2.LINE_AA)
        elif fig_type == 'triangle':
            pts = np.array([[x, y - tamaño], [x - tamaño, y + tamaño], [x + tamaño, y + tamaño]], np.int32)
            cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
        elif fig_type == 'star':
            self._dibujar_estrella_mini(frame, (x, y), tamaño, color)
        elif fig_type == 'heart':
            self._dibujar_corazon_mini(frame, (x, y), tamaño, color)

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
        """Dibuja un corazón miniatura"""
        x, y = center
        # Simple heart shape
        cv2.ellipse(frame, (x - size // 2, y - size // 3), (size // 2, size // 3), 0, 0, 180, color, 1, cv2.LINE_AA)
        cv2.ellipse(frame, (x + size // 2, y - size // 3), (size // 2, size // 3), 0, 0, 180, color, 1, cv2.LINE_AA)
        pts = np.array([[x, y + size // 2], [x - size, y - size // 3], [x + size, y - size // 3]], np.int32)
        cv2.polylines(frame, [pts], True, color, 1, cv2.LINE_AA)

    def dibujar_menu_principal(self, frame):
        """Menú principal optimizado"""
        frame[:] = self.colores['fondo']
        self.dibujar_grid_minimal(frame, spacing=100)  # More spaced grid
        self.dibujar_borde_esquinas(frame)

        # Título
        titulo = "[ NEURAL CANVAS v2.1 ]"
        x_titulo = self.ancho // 2 - 200
        self.dibujar_texto_limpio(frame, titulo, (x_titulo, 100),
                                  1.2, self.colores['azul_electrico'], 2)

        # Línea separadora
        cv2.line(frame, (x_titulo - 20, 120), (x_titulo + 440, 120),
                 self.colores['azul_oscuro'], 1, cv2.LINE_AA)

        # Items del menú
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

        # Footer
        footer_y = self.alto - 40
        self.dibujar_texto_limpio(frame, "COMPUTER VISION LAB | 2025",
                                  (self.ancho // 2 - 180, footer_y),
                                  0.6, self.colores['gris_tech'], 1)

        return frame

    def dibujar_hud_superior(self, frame, texto_modo):
        """HUD superior optimizado"""
        cv2.rectangle(frame, (0, 0), (self.ancho, 45), self.colores['panel'], -1)

        self.dibujar_texto_limpio(frame, f">> {texto_modo}", (20, 30),
                                  0.6, self.colores['azul_electrico'], 1)

        fps_text = f"FPS: {self.fps_actual}"
        color_fps = self.colores['verde_matrix'] if self.fps_actual > 25 else self.colores['azul_electrico']
        self.dibujar_texto_limpio(frame, fps_text, (self.ancho - 120, 30),
                                  0.5, color_fps, 1)

    def _dibujar_overlay_perf(self, frame):
        """Panel ligero con métricas de performance (toggle con tecla F)."""

        panel_x, panel_y = 15, 60
        ancho, alto = 210, 120
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + ancho, panel_y + alto), (15, 15, 20), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + ancho, panel_y + alto), (60, 100, 180), 1)

        lineas = [
            f"total : {self.perf_metrics['total']*1000:.1f} ms",
            f"detect: {self.perf_metrics['detect']*1000:.1f} ms",
            f"logic : {self.perf_metrics['handle']*1000:.1f} ms",
            f"figures: {self.perf_metrics['draw_figures']*1000:.1f} ms",
            f"menu up: {self.perf_metrics['menu_update']*1000:.1f} ms",
            f"menu dr: {self.perf_metrics['menu_draw']*1000:.1f} ms",
        ]

        for i, texto in enumerate(lineas):
            self.dibujar_texto_limpio(frame, texto, (panel_x + 10, panel_y + 25 + i * 18), 0.45,
                                      self.colores['azul_electrico'], 1)

    def modo_seguimiento_color(self, frame):
        """Modo tracking optimizado"""
        self.dibujar_grid_minimal(frame)
        self.dibujar_borde_esquinas(frame)
        self.dibujar_hud_superior(frame, "COLOR TRACKING MODULE")

        frame_procesado = self.color_painter.process_frame(frame)
        return frame_procesado

    def modo_figuras_gestos(self, frame):
        """Modo gestos con display de selección"""
        self.dibujar_grid_minimal(frame)
        self.dibujar_borde_esquinas(frame)
        self.dibujar_hud_superior(frame, "GESTURE RECOGNITION MODULE")

        # Mostrar display de selección si hay figura seleccionada
        self.dibujar_display_seleccion(frame)
        now = time.perf_counter()
        dt = now - self.ultimo_dt
        self.ultimo_dt = now

        # Evitar rotación cuando el menú está activo
        menu_activo = self.neon_menu.is_visible()
        self.gesture_3d.set_rotation_enabled(not menu_activo)
        self.gesture_3d.set_external_menu_active(menu_activo)

        profile = {} if self.debug_perf else None
        frame_procesado = self.gesture_3d.process_frame(frame, profile=profile)

        update_start = time.perf_counter()
        self._actualizar_neon_menu(dt)
        menu_activo = self.neon_menu.is_visible()
        self.gesture_3d.set_rotation_enabled(not menu_activo)
        self.gesture_3d.set_external_menu_active(menu_activo)
        menu_update_dt = time.perf_counter() - update_start

        menu_draw_start = time.perf_counter()
        self.neon_menu.draw(frame_procesado)
        menu_draw_dt = time.perf_counter() - menu_draw_start

        if self.debug_perf:
            self.perf_metrics = {
                "total": time.perf_counter() - now,
                "detect": profile.get("detect", 0.0) if profile else 0.0,
                "handle": profile.get("handle", 0.0) if profile else 0.0,
                "draw_figures": profile.get("draw_figures", 0.0) if profile else 0.0,
                "menu_update": menu_update_dt,
                "menu_draw": menu_draw_dt,
            }
            self._dibujar_overlay_perf(frame_procesado)

        return frame_procesado

    def procesar_teclas(self, key):
        """Procesa todas las entradas de teclado de manera centralizada"""
        # Navegación principal
        if key == ord('q'):
            print("\n[ SHUTDOWN ] Cerrando sistema...")
            return 'quit'
        elif key == ord('3') and self.modo_actual == "menu":
            self.lanzar_filtro_ar()
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

        # Controles modo color
        elif self.modo_actual == "color":
            self._procesar_teclas_color(key)

        # Controles modo gestos
        elif self.modo_actual == "gestos":
            self._procesar_teclas_gestos(key)

        return 'continue'

    def _procesar_teclas_color(self, key):
        """Procesa teclas específicas del modo color"""
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
        # Presets HSV (teclas 1-6 en modo color)
        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')]:
            preset_num = key - ord('0')
            self.color_painter.set_hsv_preset(preset_num)
        # Toggle calibracion HSV
        elif key == ord('h'):
            is_active = self.color_painter.toggle_hsv_calibration()
            state = "ACTIVADA" if is_active else "DESACTIVADA"
            print(f"[ HSV ] Calibracion {state}")
        # Reset a preset por defecto
        elif key == ord('r'):
            self.color_painter.reset_to_default_preset()
            print("[ HSV ] Reset a preset por defecto (AZUL)")

    def _procesar_teclas_gestos(self, key):
        """Procesa teclas específicas del modo gestos"""
        posicion_central = (self.ancho // 2, self.alto // 2)

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
        """Sincroniza gestos con el menú neon sin bloquear la app."""

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

    def ejecutar(self):
        """Bucle principal optimizado"""
        if not self.inicializar():
            return

        print("\n╔═══════════════════════════════════════╗")
        print("║  SISTEMA INICIADO CORRECTAMENTE      ║")
        print("║  [1] Color Tracking                  ║")
        print("║  [2] Gesture Recognition             ║")
        print("║  [M] Main Menu                       ║")
        print("║  [3] AR Filter                       ║")
        print("║  [Q] Shutdown                        ║")
        print("╚═══════════════════════════════════════╝\n")

        cv2.namedWindow('NEURAL CANVAS v2.1', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('NEURAL CANVAS v2.1', 1200, 700)

        try:
            while True:
                # Control de FPS
                self.fps_counter += 1
                if time.time() - self.fps_time >= 1.0:
                    self.fps_actual = self.fps_counter
                    self.fps_counter = 0
                    self.fps_time = time.time()

                # Captura de frame
                ret, frame = self.cap.read()
                if not ret:
                    print("[ ERROR ] Frame capture failed")
                    break

                frame = cv2.flip(frame, 1)

                # Redimensionar si es necesario
                if frame.shape[1] != self.ancho or frame.shape[0] != self.alto:
                    frame = cv2.resize(frame, (self.ancho, self.alto))

                # Procesar según modo actual
                frame_procesado = frame.copy()
                if self.modo_actual == "menu":
                    frame_procesado = self.dibujar_menu_principal(frame_procesado)
                elif self.modo_actual == "color":
                    frame_procesado = self.modo_seguimiento_color(frame_procesado)
                elif self.modo_actual == "gestos":
                    frame_procesado = self.modo_figuras_gestos(frame_procesado)

                # Mostrar frame
                cv2.imshow('NEURAL CANVAS v2.1', frame_procesado)

                # Procesar entrada de teclado
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Si hay tecla presionada
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
        print("\n╔═══════════════════════════════════════╗")
        print("║  SISTEMA APAGADO CORRECTAMENTE       ║")
        print("╚═══════════════════════════════════════╝")


if __name__ == "__main__":
    app = PizarraNeon()
    app.ejecutar()
