import cv2
import numpy as np
import math
import time


class NeonEffects:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.time_start = time.time()

    def draw_glowing_text(self, image, text, position, font_scale, color, thickness=2, glow_intensity=1):
        """Dibujar texto con glow MINIMALISTA - estilo hacker"""
        # Solo UNA capa de glow sutil
        if glow_intensity > 0:
            glow_color = (
                min(255, int(color[0] * 0.6)),
                min(255, int(color[1] * 0.6)),
                min(255, int(color[2] * 0.6))
            )
            # Blur sutil en lugar de múltiples capas
            offset = 1
            cv2.putText(image, text, (position[0] + offset, position[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, glow_color, thickness + 1)
            cv2.putText(image, text, (position[0] - offset, position[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, glow_color, thickness + 1)

        # Texto principal con anti-aliasing
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)

    def draw_cyber_grid(self, image, spacing=50, pulse_speed=1.5):
        """Grid minimalista estilo Matrix/Hacker - AZULES"""
        current_time = time.time() - self.time_start
        pulse = (math.sin(current_time * pulse_speed) + 1) / 2

        # Colores azules más oscuros y sutiles
        base_intensity = 25 + int(pulse * 15)
        grid_color = (base_intensity + 10, base_intensity, base_intensity // 2)  # Tonos azul oscuro

        # Líneas más espaciadas y sutiles
        for x in range(0, self.width, spacing):
            alpha = 0.15 if x % (spacing * 3) == 0 else 0.08
            overlay = image.copy()
            cv2.line(overlay, (x, 0), (x, self.height), grid_color, 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        for y in range(0, self.height, spacing):
            alpha = 0.15 if y % (spacing * 3) == 0 else 0.08
            overlay = image.copy()
            cv2.line(overlay, (0, y), (self.width, y), grid_color, 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    def draw_matrix_rain(self, image, density=20):
        """Efecto de 'lluvia' de Matrix sutil en el fondo"""
        current_time = time.time() - self.time_start

        for i in range(density):
            x = int((i * 137.5) % self.width)  # Distribución pseudo-aleatoria
            y = int((current_time * 50 + i * 50) % self.height)

            alpha = 0.1 + 0.05 * math.sin(current_time + i)
            color = (0, int(100 + 50 * math.sin(current_time + i)), 0)

            overlay = image.copy()
            cv2.circle(overlay, (x, y), 1, color, -1)
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    def draw_pulsating_border(self, image, color, thickness=2, speed=2.0):
        """Borde minimalista estilo terminal - esquinas biseladas"""
        current_time = time.time() - self.time_start
        pulse = (math.sin(current_time * speed) + 1) / 2

        # Color con pulso sutil
        border_color = (
            int(color[0] * (0.6 + pulse * 0.4)),
            int(color[1] * (0.6 + pulse * 0.4)),
            int(color[2] * (0.6 + pulse * 0.4))
        )

        margin = 15
        corner_len = 40

        # Esquinas superiores
        cv2.line(image, (margin, margin + corner_len), (margin, margin),
                 border_color, thickness, cv2.LINE_AA)
        cv2.line(image, (margin, margin), (margin + corner_len, margin),
                 border_color, thickness, cv2.LINE_AA)

        cv2.line(image, (self.width - margin - corner_len, margin),
                 (self.width - margin, margin), border_color, thickness, cv2.LINE_AA)
        cv2.line(image, (self.width - margin, margin),
                 (self.width - margin, margin + corner_len), border_color, thickness, cv2.LINE_AA)

        # Esquinas inferiores
        cv2.line(image, (margin, self.height - margin - corner_len),
                 (margin, self.height - margin), border_color, thickness, cv2.LINE_AA)
        cv2.line(image, (margin, self.height - margin),
                 (margin + corner_len, self.height - margin), border_color, thickness, cv2.LINE_AA)

        cv2.line(image, (self.width - margin - corner_len, self.height - margin),
                 (self.width - margin, self.height - margin), border_color, thickness, cv2.LINE_AA)
        cv2.line(image, (self.width - margin, self.height - margin - corner_len),
                 (self.width - margin, self.height - margin), border_color, thickness, cv2.LINE_AA)

        # Puntos en las esquinas
        cv2.circle(image, (margin, margin), 3, border_color, -1)
        cv2.circle(image, (self.width - margin, margin), 3, border_color, -1)
        cv2.circle(image, (margin, self.height - margin), 3, border_color, -1)
        cv2.circle(image, (self.width - margin, self.height - margin), 3, border_color, -1)

    def create_scan_lines(self, image, line_spacing=3, speed=0.5):
        """Líneas de escaneo CRT muy sutiles"""
        current_time = time.time() - self.time_start
        offset = int(current_time * speed * 10) % line_spacing

        overlay = image.copy()
        for y in range(offset, self.height, line_spacing):
            cv2.line(overlay, (0, y), (self.width, y), (0, 0, 0), 1)

        # Efecto MUY sutil
        cv2.addWeighted(overlay, 0.05, image, 0.95, 0, image)

    def draw_terminal_cursor(self, image, x, y, size=10, blink_speed=2.0):
        """Cursor parpadeante estilo terminal"""
        current_time = time.time() - self.time_start
        if math.sin(current_time * blink_speed) > 0:
            color = (0, 180, 255)  # Azul cibernético
            cv2.rectangle(image, (x, y - size), (x + 3, y), color, -1)

    def draw_hex_pattern(self, image, hex_size=60, alpha=0.03):
        """Patrón hexagonal muy sutil en el fondo"""
        current_time = time.time() - self.time_start

        for x in range(-hex_size, self.width + hex_size, hex_size):
            for y in range(-hex_size, self.height + hex_size, int(hex_size * 0.866)):
                offset_x = hex_size // 2 if (y // int(hex_size * 0.866)) % 2 else 0
                hex_x = x + offset_x
                hex_y = y

                if 0 <= hex_x < self.width and 0 <= hex_y < self.height:
                    points = []
                    for i in range(6):
                        angle = 2 * math.pi / 6 * i + math.pi / 6
                        px = hex_x + hex_size * 0.3 * math.cos(angle)
                        py = hex_y + hex_size * 0.3 * math.sin(angle)
                        points.append((int(px), int(py)))

                    overlay = image.copy()
                    color = (40, 25, 15)  # Azul muy oscuro
                    cv2.polylines(overlay, [np.array(points, np.int32)], True, color, 1)
                    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    def draw_data_stream(self, image, x, y, length=5, color=(0, 150, 255)):
        """Pequeñas líneas de 'datos' animadas"""
        current_time = time.time() - self.time_start
        alpha = 0.3 + 0.2 * math.sin(current_time * 3)

        overlay = image.copy()
        for i in range(length):
            y_pos = y + i * 3
            if y_pos < self.height:
                cv2.line(overlay, (x, y_pos), (x + 2, y_pos), color, 1)

        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)