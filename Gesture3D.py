import cv2
import math
import time
import numpy as np
from enum import Enum
from geometry_utils import rotate_points


class Gesture(Enum):
    NONE = 0
    FIST = 1
    OPEN_HAND = 2
    PINCH = 3
    VICTORY = 4


class MenuState(Enum):
    HIDDEN = 0
    VISIBLE = 1


class SelectionMode(Enum):
    NORMAL = 0
    SCALE = 1


class Gesture3D:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        # Estado de figuras
        self.figures = []
        self.selected_figure = None

        # Estado de gestos
        self.current_gesture = Gesture.NONE
        self.pinch_active = False
        self.last_pinch_position = None
        self.pinch_start_position = None

        # Coordenadas de dedos para escala
        self.thumb_tip = None
        self.index_tip = None
        self.current_finger_distance = 0

        # Menú mejorado con más figuras
        self.menu_state = MenuState.HIDDEN
        self.menu_position = (width // 2, height // 2)
        self.menu_radius = 200

        # Figuras disponibles en el menú
        self.available_figures = [
            'circle', 'square', 'triangle', 'star', 'heart', 'hexagon'
        ]

        # Modos de selección
        self.selection_mode = SelectionMode.NORMAL
        self.scale_reference_distance = 0
        self.scale_original_size = 0

        # Parámetros de escala mejorados
        self.min_finger_distance = 20
        self.max_finger_distance = 250
        self.min_figure_size = 15
        self.max_figure_size = 300

        # Suavizado de animaciones
        self.size_smoothing = 0.2
        self.previous_sizes = {}

        # Configuración de colores
        self.colors = [
            (255, 120, 0),  # Azul eléctrico
            (255, 200, 100),  # Azul claro
            (80, 255, 0),  # Verde Matrix
            (255, 255, 80),  # Cyan
            (0, 140, 255),  # Naranja
            (255, 50, 180),  # Rosa
            (180, 255, 120),  # Verde claro
            (120, 180, 255)  # Azul violeta
        ]
        self.current_color_index = 0

        # MediaPipe
        self.mediapipe_available = False
        self._initialize_mediapipe()

        # Para suavizar cambios de gesto
        self.last_victory_time = 0.0
        self.last_open_hand_time = 0.0
        self.gesture_cooldown = 0.5

        # Rotacion continua
        self.last_frame_time = time.perf_counter()
        self.rotation_speed = math.pi  # rad/s (180 grados por segundo)

    def _initialize_mediapipe(self):
        """Inicializa MediaPipe de forma segura"""
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.mediapipe_available = True
            print("[INFO] MediaPipe inicializado correctamente")
        except ImportError:
            print("[WARN] MediaPipe no disponible, solo interfaz sin gestos.")
            self.mediapipe_available = False

    def _get_pixel_landmarks(self, hand_landmarks):
        """Convierte landmarks normalizados a coordenadas de píxel"""
        return [(int(lm.x * self.width), int(lm.y * self.height))
                for lm in hand_landmarks.landmark]

    def _get_finger_states(self, landmarks_px, handedness_label):
        """Detección optimizada de dedos extendidos"""
        fingers = [0, 0, 0, 0, 0]

        # Pulgar (lógica mejorada)
        thumb_tip = landmarks_px[4]
        thumb_ip = landmarks_px[3]
        thumb_mcp = landmarks_px[2]

        if handedness_label == "Right":
            fingers[0] = 1 if thumb_tip[0] < thumb_ip[0] < thumb_mcp[0] else 0
        else:
            fingers[0] = 1 if thumb_tip[0] > thumb_ip[0] > thumb_mcp[0] else 0

        # Índice, medio, anular, meñique
        tip_ids = [8, 12, 16, 20]
        pip_ids = [6, 10, 14, 18]

        for i, (tip_id, pip_id) in enumerate(zip(tip_ids, pip_ids), start=1):
            if landmarks_px[tip_id][1] < landmarks_px[pip_id][1]:
                fingers[i] = 1

        return fingers

    def detect_gestures(self, frame):
        """Detección de gestos optimizada"""
        if not self.mediapipe_available:
            return Gesture.NONE, None, None

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if not results.multi_hand_landmarks:
                self.thumb_tip = None
                self.index_tip = None
                self.current_finger_distance = 0
                return Gesture.NONE, None, None

            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks_px = self._get_pixel_landmarks(hand_landmarks)

            # Capturar posiciones de dedos
            self.thumb_tip = landmarks_px[4]
            self.index_tip = landmarks_px[8]

            # Handedness
            handedness_label = "Right"
            if results.multi_handedness:
                handedness_label = results.multi_handedness[0].classification[0].label

            # Estados de los dedos
            fingers = self._get_finger_states(landmarks_px, handedness_label)
            extended_count = sum(fingers)

            # Punto de pinch
            pinch_position = (
                (self.thumb_tip[0] + self.index_tip[0]) // 2,
                (self.thumb_tip[1] + self.index_tip[1]) // 2
            )

            # Calcular distancia entre dedos
            self.current_finger_distance = math.hypot(
                self.thumb_tip[0] - self.index_tip[0],
                self.thumb_tip[1] - self.index_tip[1]
            )

            # Umbral de pinch adaptativo
            xs = [p[0] for p in landmarks_px]
            hand_width = max(1, max(xs) - min(xs))
            pinch_threshold = hand_width * 0.3  # Más sensible

            is_pinching = self.current_finger_distance < pinch_threshold

            # Clasificación de gesto mejorada
            gesture = Gesture.NONE

            if is_pinching:
                gesture = Gesture.PINCH
            else:
                # VICTORY: solo índice y medio extendidos
                if fingers[1] == 1 and fingers[2] == 1 and sum(fingers) == 2:
                    gesture = Gesture.VICTORY
                elif extended_count == 0:
                    gesture = Gesture.FIST
                elif extended_count >= 4:
                    gesture = Gesture.OPEN_HAND

            return gesture, pinch_position, hand_landmarks

        except Exception as e:
            print(f"[ERROR] En detección de gestos: {e}")
            self.thumb_tip = None
            self.index_tip = None
            self.current_finger_distance = 0
            return Gesture.NONE, None, None

    def handle_gestures(self, gesture, pinch_position, current_time):
        """Manejo de gestos optimizado"""
        # Calcular deltaTime para rotacion continua
        now = time.perf_counter()
        dt = now - self.last_frame_time
        self.last_frame_time = now

        # Menú con VICTORY (con cooldown)
        if (gesture == Gesture.VICTORY and
                current_time - self.last_victory_time > self.gesture_cooldown):
            self.menu_state = MenuState.VISIBLE if self.menu_state == MenuState.HIDDEN else MenuState.HIDDEN
            if pinch_position:
                self.menu_position = pinch_position
            self.last_victory_time = current_time

        # PINCH para selección/creación/movimiento/escala
        elif gesture == Gesture.PINCH:
            if not self.pinch_active:
                # Comienza nuevo PINCH
                self.pinch_active = True
                self.last_pinch_position = pinch_position
                self.pinch_start_position = pinch_position

                if self.menu_state == MenuState.VISIBLE:
                    self.handle_menu_selection(pinch_position)
                else:
                    if self.selection_mode == SelectionMode.SCALE and self.selected_figure:
                        # Iniciar modo escala
                        self.scale_reference_distance = self.current_finger_distance
                        self.scale_original_size = self.selected_figure['size']
                    else:
                        self.handle_figure_selection(pinch_position)
            else:
                # PINCH continuo
                if self.selected_figure and pinch_position:
                    if self.selection_mode == SelectionMode.SCALE:
                        self.handle_figure_scaling_by_fingers()
                    else:
                        self.move_figure(pinch_position)
                        self.last_pinch_position = pinch_position
        else:
            # PINCH liberado
            self.pinch_active = False
            self.pinch_start_position = None

        # Rotacion continua con mano abierta (MUTEX: no rotar si PINCH activo)
        if (gesture == Gesture.OPEN_HAND and
                self.selected_figure and
                not self.pinch_active):
            self.rotate_figure_continuous(dt)

    def handle_figure_scaling_by_fingers(self):
        """Escala suavizada basada en distancia entre dedos"""
        if not self.selected_figure or self.scale_reference_distance == 0:
            return

        if self.current_finger_distance == 0:
            return

        # Calcular factor de escala
        scale_factor = self.current_finger_distance / self.scale_reference_distance

        # Aplicar función de suavizado
        smooth_scale = self._apply_smoothing(scale_factor)

        # Calcular nuevo tamaño
        new_size = int(self.scale_original_size * smooth_scale)
        new_size = max(self.min_figure_size, min(self.max_figure_size, new_size))

        # Suavizar transición de tamaño
        figure_id = id(self.selected_figure)
        if figure_id in self.previous_sizes:
            current_size = self.selected_figure['size']
            smoothed_size = int(current_size + (new_size - current_size) * self.size_smoothing)
            self.selected_figure['size'] = smoothed_size
        else:
            self.selected_figure['size'] = new_size

        self.previous_sizes[figure_id] = self.selected_figure['size']

    def _apply_smoothing(self, scale_factor):
        """Aplica suavizado al factor de escala"""
        # Función de easing para transiciones más suaves
        if scale_factor > 1:
            return 1 + (scale_factor - 1) * 0.7  # Más suave al agrandar
        else:
            return 1 - (1 - scale_factor) * 0.5  # Más suave al reducir

    def handle_menu_selection(self, position):
        """Detección de ítem del menú mejorada"""
        if not position:
            return

        # Crear items del menú en círculo
        menu_items = []
        item_count = len(self.available_figures)

        for i, fig_type in enumerate(self.available_figures):
            angle = 2 * math.pi * i / item_count
            x = self.menu_position[0] + int(self.menu_radius * 0.6 * math.cos(angle))
            y = self.menu_position[1] + int(self.menu_radius * 0.6 * math.sin(angle))

            menu_items.append({
                'type': fig_type,
                'pos': (x, y),
                'size': 35
            })

        # Añadir botones de control en el centro
        menu_items.extend([
            {'type': 'color', 'pos': (self.menu_position[0], self.menu_position[1] - 30), 'size': 25},
            {'type': 'delete', 'pos': (self.menu_position[0], self.menu_position[1] + 30), 'size': 25}
        ])

        for item in menu_items:
            distance = math.hypot(
                position[0] - item['pos'][0],
                position[1] - item['pos'][1]
            )

            if distance < item['size']:
                if item['type'] == 'color':
                    self.current_color_index = (self.current_color_index + 1) % len(self.colors)
                    print(f"[MENU] Color cambiado a índice: {self.current_color_index}")
                elif item['type'] == 'delete':
                    self.delete_selected_figure()
                else:
                    self.create_figure(item['type'], position)

                self.menu_state = MenuState.HIDDEN
                break

    def handle_figure_selection(self, position):
        """Selección de figura optimizada"""
        if not position:
            return

        # Buscar figura más cercana
        closest_figure = None
        min_distance = float('inf')

        for figure in self.figures:
            fx, fy = figure['position']
            distance = math.hypot(fx - position[0], fy - position[1])

            if distance < figure['size'] * 1.2 and distance < min_distance:  # Margen aumentado
                min_distance = distance
                closest_figure = figure

        if closest_figure:
            self.selected_figure = closest_figure
            selection_color = (0, 0, 255) if self.selection_mode == SelectionMode.SCALE else (255, 255, 80)
            self.selected_figure['selection_color'] = selection_color

            mode_text = "ESCALA" if self.selection_mode == SelectionMode.SCALE else "NORMAL"
            print(f"[SELECTION] Figura {closest_figure['type']} seleccionada en modo {mode_text}")
        else:
            # Crear nueva figura si no hay selección
            self.create_figure('circle', position)

    def create_figure(self, figure_type, position):
        """Crear figura con propiedades extendidas"""
        figure = {
            'type': figure_type,
            'position': position,
            'size': 60,  # Tamaño inicial aumentado
            'color': self.colors[self.current_color_index],
            'rotation': 0.0,  # Radianes
            'selection_color': (255, 255, 80),
            'creation_time': time.time(),
            'id': len(self.figures)  # ID único
        }
        self.figures.append(figure)
        self.selected_figure = figure
        print(f"[CREATE] Nueva figura {figure_type} creada")
        return figure

    def create_figure_by_key(self, figure_type, position=None):
        """Crear figura con tecla"""
        if position is None:
            position = self.last_pinch_position or (self.width // 2, self.height // 2)
        return self.create_figure(figure_type, position)

    def move_figure(self, new_position):
        """Mover figura con límites mejorados"""
        if self.selected_figure and new_position:
            size = self.selected_figure['size']
            margin = size + 10  # Margen de seguridad
            x = max(margin, min(self.width - margin, new_position[0]))
            y = max(margin, min(self.height - margin, new_position[1]))
            self.selected_figure['position'] = (x, y)

    def rotate_figure(self):
        """Rotar figura seleccionada (legacy, incremento fijo)"""
        if self.selected_figure:
            self.selected_figure['rotation'] = (self.selected_figure['rotation'] + 6) % 360

    def rotate_figure_continuous(self, dt: float):
        """Rotar figura seleccionada de forma continua basada en deltaTime."""
        if self.selected_figure:
            delta_angle = self.rotation_speed * dt
            self.selected_figure['rotation'] += delta_angle  # Más lento

    def delete_selected_figure(self):
        """Eliminar figura seleccionada"""
        if self.selected_figure and self.selected_figure in self.figures:
            fig_type = self.selected_figure['type']
            self.figures.remove(self.selected_figure)
            self.selected_figure = None
            print(f"[DELETE] Figura {fig_type} eliminada")

    def clear_figures(self):
        """Limpiar todas las figuras"""
        figure_count = len(self.figures)
        self.figures.clear()
        self.selected_figure = None
        print(f"[CLEAR] {figure_count} figuras eliminadas")

    def toggle_scale_mode(self):
        """Cambiar entre modo normal y modo escala"""
        if self.selection_mode == SelectionMode.NORMAL:
            self.selection_mode = SelectionMode.SCALE
            print("[MODE] Cambiado a modo ESCALA")
        else:
            self.selection_mode = SelectionMode.NORMAL
            print("[MODE] Cambiado a modo NORMAL")

        # Actualizar color de selección
        if self.selected_figure:
            selection_color = (0, 0, 255) if self.selection_mode == SelectionMode.SCALE else (255, 255, 80)
            self.selected_figure['selection_color'] = selection_color

    def draw_finger_connection_line(self, frame):
        """Dibujar línea entre dedos en modo escala"""
        if not self.thumb_tip or not self.index_tip:
            return

        # Color dinámico basado en distancia
        distance_ratio = self.current_finger_distance / self.max_finger_distance
        color = (
            int(255 * (1 - distance_ratio)),
            int(255 * distance_ratio),
            0
        )

        # Línea entre dedos
        cv2.line(frame, self.thumb_tip, self.index_tip, color, 3, cv2.LINE_AA)

        # Círculos en las puntas
        cv2.circle(frame, self.thumb_tip, 8, (255, 100, 100), -1, cv2.LINE_AA)
        cv2.circle(frame, self.index_tip, 8, (100, 100, 255), -1, cv2.LINE_AA)

    def draw_enhanced_menu(self, frame):
        """Dibujar menú circular mejorado"""
        center = self.menu_position

        # Fondo del menú
        overlay = frame.copy()
        cv2.circle(overlay, center, self.menu_radius, (25, 25, 40), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        cv2.circle(frame, center, self.menu_radius, (255, 220, 150), 2, cv2.LINE_AA)

        # Dibujar figuras alrededor del círculo
        item_count = len(self.available_figures)
        for i, fig_type in enumerate(self.available_figures):
            angle = 2 * math.pi * i / item_count
            x = center[0] + int(self.menu_radius * 0.6 * math.cos(angle))
            y = center[1] + int(self.menu_radius * 0.6 * math.sin(angle))

            # Dibujar botón de figura
            self._draw_figure_menu_button(frame, (x, y), fig_type)

        # Botones de control centrales
        self._draw_control_buttons(frame, center)

    def _draw_figure_menu_button(self, frame, pos, fig_type):
        """Dibujar botón de figura en el menú"""
        x, y = pos

        # Fondo del botón
        cv2.circle(frame, pos, 30, (50, 50, 70), -1, cv2.LINE_AA)
        cv2.circle(frame, pos, 30, (200, 200, 220), 2, cv2.LINE_AA)

        # Dibujar figura miniatura
        color = (200, 200, 255)
        if fig_type == 'circle':
            cv2.circle(frame, pos, 15, color, 2, cv2.LINE_AA)
        elif fig_type == 'square':
            cv2.rectangle(frame, (x - 12, y - 12), (x + 12, y + 12), color, 2, cv2.LINE_AA)
        elif fig_type == 'triangle':
            pts = np.array([[x, y - 12], [x - 12, y + 8], [x + 12, y + 8]], np.int32)
            cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
        elif fig_type == 'star':
            self._draw_star(frame, pos, 12, color, 2)
        elif fig_type == 'heart':
            self._draw_heart(frame, pos, 10, color, 2)
        elif fig_type == 'hexagon':
            self._draw_hexagon(frame, pos, 12, color, 2)

        # Texto
        text = fig_type[:3].upper()
        cv2.putText(frame, text, (x - 15, y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)

    def _draw_control_buttons(self, frame, center):
        """Dibujar botones de control en el centro del menú"""
        x, y = center

        # Botón de color
        cv2.circle(frame, (x, y - 30), 20, (60, 60, 80), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y - 30), 20, (255, 255, 255), 2, cv2.LINE_AA)
        next_color = self.colors[(self.current_color_index + 1) % len(self.colors)]
        cv2.circle(frame, (x, y - 30), 15, next_color, -1, cv2.LINE_AA)
        cv2.putText(frame, "COLOR", (x - 20, y - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)

        # Botón de eliminar
        cv2.circle(frame, (x, y + 30), 20, (60, 60, 80), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y + 30), 20, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (x - 12, y + 30 - 12), (x + 12, y + 30 + 12), (255, 100, 100), 3, cv2.LINE_AA)
        cv2.line(frame, (x - 12, y + 30 + 12), (x + 12, y + 30 - 12), (255, 100, 100), 3, cv2.LINE_AA)
        cv2.putText(frame, "DEL", (x - 12, y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)

    def _draw_star(self, frame, center, size, color, thickness):
        """Dibujar una estrella"""
        x, y = center
        points = []
        for i in range(10):
            angle = math.pi / 2 + i * math.pi / 5
            r = size if i % 2 == 0 else size / 2
            points.append((int(x + r * math.cos(angle)), int(y + r * math.sin(angle))))

        pts = np.array(points, np.int32)
        cv2.polylines(frame, [pts], True, color, thickness, cv2.LINE_AA)

    def _draw_heart(self, frame, center, size, color, thickness):
        """Dibujar un corazón"""
        x, y = center
        # Semi-círculos superiores
        cv2.ellipse(frame, (x - size // 2, y - size // 3), (size // 2, size // 3), 0, 0, 180, color, thickness,
                    cv2.LINE_AA)
        cv2.ellipse(frame, (x + size // 2, y - size // 3), (size // 2, size // 3), 0, 0, 180, color, thickness,
                    cv2.LINE_AA)
        # Triángulo inferior
        pts = np.array([[x, y + size // 2], [x - size, y - size // 3], [x + size, y - size // 3]], np.int32)
        cv2.polylines(frame, [pts], True, color, thickness, cv2.LINE_AA)

    def _draw_hexagon(self, frame, center, size, color, thickness):
        """Dibujar un hexágono"""
        x, y = center
        points = []
        for i in range(6):
            angle = math.pi / 6 + i * math.pi / 3
            points.append((int(x + size * math.cos(angle)), int(y + size * math.sin(angle))))

        pts = np.array(points, np.int32)
        cv2.polylines(frame, [pts], True, color, thickness, cv2.LINE_AA)

    def _draw_star_rotated(self, frame, center, size, color, thickness, angle_rad):
        """Dibujar una estrella con rotacion 2D real."""
        x, y = center
        points = []
        for i in range(10):
            base_angle = math.pi / 2 + i * math.pi / 5
            r = size if i % 2 == 0 else size / 2
            points.append([x + r * math.cos(base_angle), y + r * math.sin(base_angle)])

        pts = np.array(points, dtype=np.float64)
        rotated_pts = rotate_points(pts, center, angle_rad)
        cv2.polylines(frame, [rotated_pts], True, color, thickness, cv2.LINE_AA)

    def _draw_heart_rotated(self, frame, center, size, color, thickness, angle_rad):
        """Dibujar un corazon con rotacion 2D real (aproximado con poligono)."""
        x, y = center
        points = []

        # Generar puntos del corazon como curva parametrica
        num_points = 30
        for i in range(num_points):
            t = 2 * math.pi * i / num_points
            # Ecuacion parametrica del corazon
            px = 16 * (math.sin(t) ** 3)
            py = -(13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t))
            # Escalar y centrar
            scale = size / 18.0
            points.append([x + px * scale, y + py * scale])

        pts = np.array(points, dtype=np.float64)
        rotated_pts = rotate_points(pts, center, angle_rad)
        cv2.polylines(frame, [rotated_pts], True, color, thickness, cv2.LINE_AA)

    def _draw_hexagon_rotated(self, frame, center, size, color, thickness, angle_rad):
        """Dibujar un hexagono con rotacion 2D real."""
        x, y = center
        points = []
        for i in range(6):
            base_angle = math.pi / 6 + i * math.pi / 3
            points.append([x + size * math.cos(base_angle), y + size * math.sin(base_angle)])

        pts = np.array(points, dtype=np.float64)
        rotated_pts = rotate_points(pts, center, angle_rad)
        cv2.polylines(frame, [rotated_pts], True, color, thickness, cv2.LINE_AA)

    def draw_figures(self, frame):
        """Dibujar todas las figuras optimizado"""
        for figure in self.figures:
            self._draw_single_figure(frame, figure)

    def _draw_single_figure(self, frame, figure):
        """Dibujar una figura individual con rotacion 2D real."""
        color = figure['color']
        pos = figure['position']
        size = figure['size']
        angle_rad = figure['rotation']  # Ya en radianes

        # Resaltar figura seleccionada
        if figure is self.selected_figure:
            selection_color = figure.get('selection_color', (255, 255, 80))
            cv2.circle(frame, pos, size + 8, selection_color, 2, cv2.LINE_AA)

        # Dibujar figura segun tipo CON ROTACION REAL
        if figure['type'] == 'circle':
            # Circulo no necesita rotacion visual, pero dibujamos indicador
            cv2.circle(frame, pos, size, color, 3, cv2.LINE_AA)
        elif figure['type'] == 'square':
            # Generar vertices del cuadrado
            pts = np.array([
                [pos[0] - size, pos[1] - size],
                [pos[0] + size, pos[1] - size],
                [pos[0] + size, pos[1] + size],
                [pos[0] - size, pos[1] + size]
            ], dtype=np.float64)
            rotated_pts = rotate_points(pts, pos, angle_rad)
            cv2.polylines(frame, [rotated_pts], True, color, 3, cv2.LINE_AA)
        elif figure['type'] == 'triangle':
            height = int(size * 1.5)
            pts = np.array([
                [pos[0], pos[1] - height // 2],
                [pos[0] - size, pos[1] + height // 2],
                [pos[0] + size, pos[1] + height // 2]
            ], dtype=np.float64)
            rotated_pts = rotate_points(pts, pos, angle_rad)
            cv2.polylines(frame, [rotated_pts], True, color, 3, cv2.LINE_AA)
        elif figure['type'] == 'star':
            self._draw_star_rotated(frame, pos, size, color, 3, angle_rad)
        elif figure['type'] == 'heart':
            self._draw_heart_rotated(frame, pos, size, color, 3, angle_rad)
        elif figure['type'] == 'hexagon':
            self._draw_hexagon_rotated(frame, pos, size, color, 3, angle_rad)

        # Punto central
        cv2.circle(frame, pos, 3, color, -1, cv2.LINE_AA)

        # Linea indicadora de rotacion (para todas las figuras)
        if angle_rad != 0:
            end_x = int(pos[0] + size * 0.8 * math.cos(angle_rad))
            end_y = int(pos[1] + size * 0.8 * math.sin(angle_rad))
            cv2.line(frame, pos, (end_x, end_y), color, 2, cv2.LINE_AA)

    def draw_interface(self, frame, gesture, pinch_position, hand_landmarks):
        """Dibujar interfaz completa optimizada"""
        # Línea entre dedos en modo escala
        if (self.selection_mode == SelectionMode.SCALE and
                self.thumb_tip and self.index_tip):
            self.draw_finger_connection_line(frame)

        # Punto de pinch
        if pinch_position:
            color = (0, 0, 255) if self.selection_mode == SelectionMode.SCALE else (0, 255, 0)
            cv2.circle(frame, pinch_position, 6, color, -1, cv2.LINE_AA)

        # Menú
        if self.menu_state == MenuState.VISIBLE:
            self.draw_enhanced_menu(frame)

        # Figuras
        self.draw_figures(frame)

        # Landmarks de mano
        if hand_landmarks and self.mediapipe_available:
            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )

    def process_frame(self, frame):
        """Procesar frame principal optimizado"""
        current_time = time.time()

        # Detectar gestos
        gesture, pinch_position, hand_landmarks = self.detect_gestures(frame)
        self.current_gesture = gesture

        # Actualizar última posición de pinch
        if pinch_position:
            self.last_pinch_position = pinch_position

        # Manejar gestos
        self.handle_gestures(gesture, pinch_position, current_time)

        # Dibujar interfaz
        self.draw_interface(frame, gesture, pinch_position, hand_landmarks)

        return frame