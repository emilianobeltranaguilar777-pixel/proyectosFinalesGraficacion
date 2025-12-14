import cv2
import numpy as np
import math


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

    def detect_blue_object(self, frame):
        """Detección optimizada de objeto azul con cache"""
        current_time = cv2.getTickCount() / cv2.getTickFrequency()

        # Usar cache para máscaras similares
        if self.last_mask is not None and (current_time - self.mask_cache_time) < 0.1:
            mask = self.last_mask
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Rangos para azul en HSV (optimizados)
            lower_blue = np.array([100, 120, 50])
            upper_blue = np.array([130, 255, 255])

            mask = cv2.inRange(hsv, lower_blue, upper_blue)

            # Operaciones morfológicas optimizadas
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            self.last_mask = mask
            self.mask_cache_time = current_time

        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Encontrar el contorno más grande
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 300:
                # Calcular centro del contorno
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy), largest_contour

        return None, None

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
        instructions = "[SPACE]Clear [C]Color [+/-]Size [M]Menu"
        cv2.putText(frame, instructions, (20, self.height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 170), 1, cv2.LINE_AA)

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

        # Detectar objeto azul
        current_pos, contour = self.detect_blue_object(frame)

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