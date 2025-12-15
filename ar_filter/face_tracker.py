"""Lightweight MediaPipe face tracking helper."""

import mediapipe as mp


class FaceTracker:
    """Tracks a single face and returns normalized landmarks."""

    def __init__(self, max_faces: int = 1):
        self._mp_face_mesh = mp.solutions.face_mesh
        self._mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame):
        import cv2

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(rgb_frame)
        if result.multi_face_landmarks:
            return result.multi_face_landmarks[0]
        return None

    def close(self):
        self._mesh.close()
