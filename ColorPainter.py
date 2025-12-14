import cv2
import numpy as np
import math
from filters import EMAFilter


# Presets HSV para diferentes colores (H: 0-179, S: 0-255, V: 0-255)
HSV_PRESETS = {
    1: {'name': 'AZUL', 'lower': (100, 120, 50), 'upper': (130, 255, 255)},
    2: {'name': 'VERDE', 'lower': (35, 80, 50), 'upper': (85, 255, 255)},
    3: {'name': 'ROJO', 'lower': (0, 120, 70), 'upper': (10, 255, 255)},  # Rojo bajo
    4: {'name': 'AMARILLO', 'lower': (20, 100, 100), 'upper': (35, 255, 255)},
    5: {'name': 'MAGENTA', 'lower': (140, 80, 50), 'upper': (170, 255, 255)},
    6: {'name': 'CIAN', 'lower': (80, 100, 50), 'upper': (100, 255, 255)},
}


class ColorPainter:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.brush_size = 15
        self.brush_color = (255, 120, 0)  # Azul eléctrico por defecto
        self.last_pos = None

        # Paleta de colores mejorada
        self.colors = [
            (255, 120, 0),  # Azul eléctrico
            (255, 200, 100),  # Azul claro
            (80, 255, 0),  # Verde Matrix
            (255, 255, 80),  # Cyan
            (220, 220, 200),  # Blanco tech
            (255, 50, 180),  # Rosa
            (0, 140, 255),  # Naranja
            (120, 180, 255)  # Azul violeta
        ]
        self.current_color_index = 0

        # Cache para optimización
        self.last_mask = None
        self.mask_cache_time = 0

        # Efectos de pintura
        self.paint_effects = True
        self.trail_particles = []

        # Filtro EMA para suavizar centroide
        self.centroid_filter = EMAFilter(alpha=0.4)

        # Configuracion HSV para tracking de color
        self.current_preset = 1  # Azul por defecto
        self.preset_name = HSV_PRESETS[1]['name']
        self.hsv_lower = np.array(HSV_PRESETS[1]['lower'])
        self.hsv_upper = np.array(HSV_PRESETS[1]['upper'])

        # Modo calibracion HSV
        self.hsv_calibration_mode = False
        self.hsv_window_name = 'HSV Calibration'

    def set_hsv_preset(self, preset_num: int) -> bool:
        """Cambia al preset HSV indicado. Retorna True si exitoso."""
        if preset_num not in HSV_PRESETS:
            return False

        preset = HSV_PRESETS[preset_num]
        self.current_preset = preset_num
        self.preset_name = preset['name']
        self.hsv_lower = np.array(preset['lower'])
        self.hsv_upper = np.array(preset['upper'])

        # Invalidar cache de mascara
        self.last_mask = None

        print(f"[COLOR] Preset {preset_num}: {preset['name']} "
              f"HSV({preset['lower']}) - ({preset['upper']})")
        return True

    def reset_to_default_preset(self):
        """Resetea al preset por defecto (azul)."""
        self.set_hsv_preset(1)

    def toggle_hsv_calibration(self):
        """Activa/desactiva el modo de calibracion HSV con trackbars."""
        if self.hsv_calibration_mode:
            # Desactivar: guardar valores y cerrar ventana
            self._close_hsv_calibration()
        else:
            # Activar: crear ventana con trackbars
            self._open_hsv_calibration()

        return self.hsv_calibration_mode

    def _open_hsv_calibration(self):
        """Abre la ventana de calibracion HSV."""
        self.hsv_calibration_mode = True
        cv2.namedWindow(self.hsv_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.hsv_window_name, 400, 300)

        # Crear trackbars
        cv2.createTrackbar('H Low', self.hsv_window_name, int(self.hsv_lower[0]), 179, lambda x: None)
        cv2.createTrackbar('S Low', self.hsv_window_name, int(self.hsv_lower[1]), 255, lambda x: None)
        cv2.createTrackbar('V Low', self.hsv_window_name, int(self.hsv_lower[2]), 255, lambda x: None)
        cv2.createTrackbar('H High', self.hsv_window_name, int(self.hsv_upper[0]), 179, lambda x: None)
        cv2.createTrackbar('S High', self.hsv_window_name, int(self.hsv_upper[1]), 255, lambda x: None)
        cv2.createTrackbar('V High', self.hsv_window_name, int(self.hsv_upper[2]), 255, lambda x: None)

        print("[HSV] Calibracion activada - Ajusta los trackbars")

    def _close_hsv_calibration(self):
        """Cierra la ventana de calibracion y guarda valores."""
        if self.hsv_calibration_mode:
            try:
                # Leer valores finales de trackbars
                h_low = cv2.getTrackbarPos('H Low', self.hsv_window_name)
                s_low = cv2.getTrackbarPos('S Low', self.hsv_window_name)
                v_low = cv2.getTrackbarPos('V Low', self.hsv_window_name)
                h_high = cv2.getTrackbarPos('H High', self.hsv_window_name)
                s_high = cv2.getTrackbarPos('S High', self.hsv_window_name)
                v_high = cv2.getTrackbarPos('V High', self.hsv_window_name)

                self.hsv_lower = np.array([h_low, s_low, v_low])
                self.hsv_upper = np.array([h_high, s_high, v_high])
                self.preset_name = 'CUSTOM'

                cv2.destroyWindow(self.hsv_window_name)
                print(f"[HSV] Calibracion guardada: ({h_low},{s_low},{v_low}) - ({h_high},{s_high},{v_high})")
            except cv2.error:
                pass

        self.hsv_calibration_mode = False
        self.last_mask = None  # Invalidar cache

    def _update_hsv_from_trackbars(self):
        """Actualiza valores HSV desde trackbars en tiempo real."""
        if not self.hsv_calibration_mode:
            return

        try:
            h_low = cv2.getTrackbarPos('H Low', self.hsv_window_name)
            s_low = cv2.getTrackbarPos('S Low', self.hsv_window_name)
            v_low = cv2.getTrackbarPos('V Low', self.hsv_window_name)
            h_high = cv2.getTrackbarPos('H High', self.hsv_window_name)
            s_high = cv2.getTrackbarPos('S High', self.hsv_window_name)
            v_high = cv2.getTrackbarPos('V High', self.hsv_window_name)

            self.hsv_lower = np.array([h_low, s_low, v_low])
            self.hsv_upper = np.array([h_high, s_high, v_high])
            self.last_mask = None  # Invalidar cache para ver cambios en tiempo real
        except cv2.error:
            pass

    def detect_color_object(self, frame):
        """Deteccion de objeto por color HSV configurable."""
        current_time = cv2.getTickCount() / cv2.getTickFrequency()

        # Actualizar HSV desde trackbars si esta en modo calibracion
        if self.hsv_calibration_mode:
            self._update_hsv_from_trackbars()

        # Usar cache para mascaras similares (solo si no esta calibrando)
        if (self.last_mask is not None and
                (current_time - self.mask_cache_time) < 0.1 and
                not self.hsv_calibration_mode):
            mask = self.last_mask
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Usar rangos HSV configurables
            mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

            # Operaciones morfologicas optimizadas
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            self.last_mask = mask
            self.mask_cache_time = current_time

        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Encontrar el contorno mas grande
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 300:
                # Calcular centro del contorno via moments
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy), largest_contour

        return None, None

    def detect_blue_object(self, frame):
        """Alias para compatibilidad con codigo existente."""
        return self.detect_color_object(frame)

    def add_paint_effect(self, position):
        """Añade efectos de partículas al pintar"""
        if not self.paint_effects:
            return

        # Partículas de pintura
        for _ in range(3):
            angle = np.random.uniform(0, 2 * math.pi)
            speed = np.random.uniform(1, 3)
            size = np.random.randint(1, 3)
            lifetime = np.random.uniform(0.3, 0.8)

            particle = {
                'position': list(position),
                'velocity': [speed * math.cos(angle), speed * math.sin(angle)],
                'color': self.brush_color,
                'size': size,
                'lifetime': lifetime,
                'max_lifetime': lifetime
            }
            self.trail_particles.append(particle)

    def update_paint_effects(self, dt):
        """Actualiza los efectos de pintura"""
        particles_to_remove = []

        for i, particle in enumerate(self.trail_particles):
            particle['lifetime'] -= dt

            if particle['lifetime'] <= 0:
                particles_to_remove.append(i)
                continue

            # Movimiento
            particle['position'][0] += particle['velocity'][0]
            particle['position'][1] += particle['velocity'][1]

            # Fricción
            particle['velocity'][0] *= 0.9
            particle['velocity'][1] *= 0.9

        # Eliminar partículas muertas
        for i in sorted(particles_to_remove, reverse=True):
            self.trail_particles.pop(i)

    def draw_paint_effects(self, frame):
        """Dibuja los efectos de pintura"""
        for particle in self.trail_particles:
            alpha = particle['lifetime'] / particle['max_lifetime']
            size = int(particle['size'] * alpha)

            if size > 0:
                color = tuple(int(c * alpha) for c in particle['color'])
                pos = (int(particle['position'][0]), int(particle['position'][1]))
                cv2.circle(frame, pos, size, color, -1, cv2.LINE_AA)

    def draw_on_canvas(self, current_pos):
        """Dibujar en el canvas con efectos mejorados"""
        if self.last_pos and current_pos:
            # Dibujar línea suavizada
            distance = math.hypot(current_pos[0] - self.last_pos[0],
                                  current_pos[1] - self.last_pos[1])

            if distance > 2:
                cv2.line(self.canvas, self.last_pos, current_pos,
                         self.brush_color, self.brush_size, cv2.LINE_AA)

                # Efectos de partículas
                if self.paint_effects and distance < 30:
                    self.add_paint_effect(current_pos)

        # Punto central
        cv2.circle(self.canvas, current_pos, max(2, self.brush_size // 3),
                   self.brush_color, -1, cv2.LINE_AA)

        self.last_pos = current_pos

    def clear_canvas(self):
        """Limpiar el canvas"""
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.last_pos = None
        self.trail_particles.clear()
        self.centroid_filter.reset()

    def change_brush_color(self):
        """Cambiar al siguiente color de la paleta"""
        self.current_color_index = (self.current_color_index + 1) % len(self.colors)
        self.brush_color = self.colors[self.current_color_index]
        return self.brush_color

    def change_brush_size(self, increment=True):
        """Cambiar tamaño del pincel con límites"""
        if increment:
            self.brush_size = min(60, self.brush_size + 2)
        else:
            self.brush_size = max(3, self.brush_size - 2)
        return self.brush_size

    def _draw_hsv_overlay(self, frame):
        """Dibuja overlay con info del preset HSV activo."""
        # Posicion en esquina superior derecha
        x_start = self.width - 220
        y_start = 60

        # Fondo semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start - 10, y_start - 25),
                      (self.width - 10, y_start + 55), (20, 20, 30), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Borde
        cv2.rectangle(frame, (x_start - 10, y_start - 25),
                      (self.width - 10, y_start + 55), (80, 80, 100), 1, cv2.LINE_AA)

        # Nombre del preset
        preset_color = (0, 255, 255) if self.hsv_calibration_mode else (100, 255, 100)
        mode_text = "[CALIB]" if self.hsv_calibration_mode else ""
        cv2.putText(frame, f"HSV: {self.preset_name} {mode_text}", (x_start, y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, preset_color, 1, cv2.LINE_AA)

        # Valores lower
        lower_text = f"L: {self.hsv_lower[0]:3d},{self.hsv_lower[1]:3d},{self.hsv_lower[2]:3d}"
        cv2.putText(frame, lower_text, (x_start, y_start + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 200), 1, cv2.LINE_AA)

        # Valores upper
        upper_text = f"H: {self.hsv_upper[0]:3d},{self.hsv_upper[1]:3d},{self.hsv_upper[2]:3d}"
        cv2.putText(frame, upper_text, (x_start, y_start + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 200), 1, cv2.LINE_AA)

    def draw_ui_elements_mejorados(self, frame, current_pos):
        """Dibujar elementos de UI mejorados"""
        # Panel de información con efecto glassmorphism
        panel_height = 80
        overlay = frame.copy()

        # Fondo con blur (simulado)
        cv2.rectangle(overlay, (0, self.height - panel_height),
                      (self.width, self.height), (15, 15, 25), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Borde superior del panel
        cv2.line(frame, (0, self.height - panel_height),
                 (self.width, self.height - panel_height),
                 (50, 50, 80), 1, cv2.LINE_AA)

        # Información del pincel con iconos
        color_display = f"Brush: Size({self.brush_size})"
        cv2.putText(frame, color_display, (20, self.height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 200), 1, cv2.LINE_AA)

        # Muestra del color actual con efecto neon
        color_sample_pos = (150, self.height - 45)
        cv2.circle(frame, color_sample_pos, 10, self.brush_color, -1, cv2.LINE_AA)
        cv2.circle(frame, color_sample_pos, 10, (255, 255, 255), 1, cv2.LINE_AA)

        # Efecto glow alrededor del color
        for i in range(3, 0, -1):
            glow_color = tuple(int(c * 0.3) for c in self.brush_color)
            cv2.circle(frame, color_sample_pos, 10 + i, glow_color, 1, cv2.LINE_AA)

        # Instrucciones con iconos
        instructions = "[SPACE]Clear [C]Color [+/-]Size [M]Menu [1-6]HSV [H]Calib"
        cv2.putText(frame, instructions, (20, self.height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 170), 1, cv2.LINE_AA)

        # Overlay HSV info (lado derecho)
        self._draw_hsv_overlay(frame)

        # Indicador de posición mejorado
        if current_pos:
            # Círculo exterior
            cv2.circle(frame, current_pos, 8, (0, 255, 0), 2, cv2.LINE_AA)
            # Círculo interior
            cv2.circle(frame, current_pos, 3, (0, 255, 0), -1, cv2.LINE_AA)

            # Cruz de guía con efecto
            size = 12
            cv2.line(frame, (current_pos[0] - size, current_pos[1]),
                     (current_pos[0] + size, current_pos[1]), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.line(frame, (current_pos[0], current_pos[1] - size),
                     (current_pos[0], current_pos[1] + size), (0, 255, 0), 1, cv2.LINE_AA)

            # Anillo exterior pulsante
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            pulse = (math.sin(current_time * 5) + 1) / 2
            ring_size = 10 + int(pulse * 4)
            cv2.circle(frame, current_pos, ring_size, (0, 200, 0), 1, cv2.LINE_AA)

    def process_frame(self, frame):
        """Procesar frame completo con efectos mejorados"""
        # Actualizar efectos de pintura
        self.update_paint_effects(1.0 / 30.0)

        # Detectar objeto azul (centroide raw)
        raw_pos, contour = self.detect_blue_object(frame)

        # Aplicar filtro EMA para suavizar centroide
        current_pos = self.centroid_filter.update(raw_pos)

        # Dibujar en el canvas si se detecta objeto
        if current_pos:
            self.draw_on_canvas(current_pos)

            # Dibujar contorno del objeto detectado
            if contour is not None and cv2.contourArea(contour) > 500:
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 1, cv2.LINE_AA)

        # Combinar canvas con frame
        frame_with_canvas = cv2.addWeighted(frame, 0.75, self.canvas, 0.25, 0)

        # Dibujar efectos de pintura
        self.draw_paint_effects(frame_with_canvas)

        # Dibujar UI mejorada
        self.draw_ui_elements_mejorados(frame_with_canvas, current_pos)

        return frame_with_canvas