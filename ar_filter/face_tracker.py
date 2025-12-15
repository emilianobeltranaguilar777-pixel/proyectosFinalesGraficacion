"""
Face Tracker Module - Encapsulates MediaPipe FaceMesh

This module handles all face detection and landmark extraction.
It does NOT render anything - only provides data.
"""

import cv2
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class FaceTracker:
    """
    Encapsulates MediaPipe FaceMesh for face landmark detection.

    Returns normalized landmarks and bounding box only.
    No rendering, no OpenGL.
    """

    # Key landmark indices for face mesh (468 total landmarks)
    # These are the most important ones for AR filters
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263
    LEFT_EYE_INNER = 133
    RIGHT_EYE_INNER = 362
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291
    TOP_LIP = 13
    BOTTOM_LIP = 14
    FOREHEAD = 10
    LEFT_CHEEK = 234
    RIGHT_CHEEK = 454
    LEFT_EAR = 234
    RIGHT_EAR = 454
    LEFT_EYEBROW_OUTER = 70
    RIGHT_EYEBROW_OUTER = 300

    def __init__(self, max_faces: int = 1, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize the face tracker.

        Args:
            max_faces: Maximum number of faces to detect
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for landmark tracking
        """
        self.max_faces = max_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self._face_mesh = None
        self._initialized = False

        if MEDIAPIPE_AVAILABLE:
            self._init_mediapipe()

    def _init_mediapipe(self):
        """Initialize MediaPipe FaceMesh."""
        mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=self.max_faces,
            refine_landmarks=True,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self._initialized = True

    def is_available(self) -> bool:
        """Check if MediaPipe is available and initialized."""
        return MEDIAPIPE_AVAILABLE and self._initialized

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a frame and extract face landmarks.

        Args:
            frame: BGR image from OpenCV (numpy array)

        Returns:
            Dictionary with:
                - 'detected': bool - whether a face was detected
                - 'landmarks': list of (x, y, z) normalized coordinates (0-1)
                - 'bbox': dict with 'x', 'y', 'width', 'height' (normalized)
                - 'frame_size': (width, height) of the input frame
        """
        result = {
            'detected': False,
            'landmarks': [],
            'bbox': None,
            'frame_size': (frame.shape[1], frame.shape[0])
        }

        if not self.is_available():
            return result

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # Process with FaceMesh
        results = self._face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return result

        # Get first face
        face_landmarks = results.multi_face_landmarks[0]

        # Extract all landmarks as normalized coordinates
        landmarks = []
        min_x, min_y = 1.0, 1.0
        max_x, max_y = 0.0, 0.0

        for landmark in face_landmarks.landmark:
            x, y, z = landmark.x, landmark.y, landmark.z
            landmarks.append((x, y, z))

            # Update bounding box
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

        result['detected'] = True
        result['landmarks'] = landmarks
        result['bbox'] = {
            'x': min_x,
            'y': min_y,
            'width': max_x - min_x,
            'height': max_y - min_y
        }

        return result

    def get_key_points(self, landmarks: list) -> dict:
        """
        Extract key facial points from landmarks.

        Args:
            landmarks: List of (x, y, z) normalized coordinates

        Returns:
            Dictionary with named key points
        """
        if not landmarks or len(landmarks) < 468:
            return {}

        return {
            'nose_tip': landmarks[self.NOSE_TIP],
            'chin': landmarks[self.CHIN],
            'left_eye_outer': landmarks[self.LEFT_EYE_OUTER],
            'right_eye_outer': landmarks[self.RIGHT_EYE_OUTER],
            'left_eye_inner': landmarks[self.LEFT_EYE_INNER],
            'right_eye_inner': landmarks[self.RIGHT_EYE_INNER],
            'left_mouth': landmarks[self.LEFT_MOUTH],
            'right_mouth': landmarks[self.RIGHT_MOUTH],
            'top_lip': landmarks[self.TOP_LIP],
            'bottom_lip': landmarks[self.BOTTOM_LIP],
            'forehead': landmarks[self.FOREHEAD],
            'left_cheek': landmarks[self.LEFT_CHEEK],
            'right_cheek': landmarks[self.RIGHT_CHEEK],
            'left_eyebrow': landmarks[self.LEFT_EYEBROW_OUTER],
            'right_eyebrow': landmarks[self.RIGHT_EYEBROW_OUTER],
        }

    def release(self):
        """Release MediaPipe resources."""
        if self._face_mesh:
            self._face_mesh.close()
            self._face_mesh = None
            self._initialized = False
