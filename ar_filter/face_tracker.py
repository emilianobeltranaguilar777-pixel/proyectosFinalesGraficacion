"""
Face tracking using MediaPipe FaceMesh.

This module ONLY handles face detection and landmark extraction.
NO rendering, NO OpenGL - just data.
"""

from typing import List, Tuple, Optional
import cv2

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None


class FaceTracker:
    """
    Wrapper for MediaPipe FaceMesh.

    Extracts normalized face landmarks from camera frames.
    Does NOT render anything - returns raw data only.
    """

    def __init__(self,
                 max_num_faces: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize FaceTracker.

        Args:
            max_num_faces: Maximum number of faces to detect
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe is not installed. Install with: pip install mediapipe")

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self._last_landmarks: Optional[List[Tuple[float, float, float]]] = None

    def process_frame(self, frame) -> Optional[List[Tuple[float, float, float]]]:
        """
        Process a BGR frame and extract face landmarks.

        Args:
            frame: OpenCV BGR image (numpy array)

        Returns:
            List of (x, y, z) normalized landmarks, or None if no face detected.
            Coordinates are normalized: x,y in [0,1], z is relative depth.
        """
        if frame is None:
            return None

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return self._last_landmarks  # Return last known landmarks for stability

        # Extract landmarks from first face
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = [
            (lm.x, lm.y, lm.z)
            for lm in face_landmarks.landmark
        ]

        self._last_landmarks = landmarks
        return landmarks

    def get_last_landmarks(self) -> Optional[List[Tuple[float, float, float]]]:
        """
        Get the most recently detected landmarks.

        Returns:
            Last detected landmarks or None if never detected.
        """
        return self._last_landmarks

    def release(self):
        """Release MediaPipe resources."""
        if self.face_mesh:
            self.face_mesh.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False


# Key landmark indices for reference
class FaceLandmarks:
    """Constants for important face landmark indices."""

    # Nose
    NOSE_TIP = 1

    # Eyes
    LEFT_EYE_INNER = 133
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263

    # Eyebrows
    LEFT_EYEBROW = 70
    RIGHT_EYEBROW = 300

    # Lips
    UPPER_LIP_CENTER = 13
    LOWER_LIP_CENTER = 14
    LEFT_MOUTH_CORNER = 61
    RIGHT_MOUTH_CORNER = 291

    # Face boundary (for face width)
    LEFT_CHEEK = 234
    RIGHT_CHEEK = 454

    # Forehead (for halo positioning)
    FOREHEAD_CENTER = 10

    # Chin
    CHIN = 152


def landmarks_to_screen(landmarks: List[Tuple[float, float, float]],
                        width: int,
                        height: int,
                        indices: Optional[List[int]] = None
                        ) -> List[Tuple[int, int]]:
    """
    Convert normalized landmarks to screen pixel coordinates.

    Args:
        landmarks: Normalized landmarks from FaceTracker
        width: Screen/frame width in pixels
        height: Screen/frame height in pixels
        indices: Optional list of landmark indices to convert (all if None)

    Returns:
        List of (x, y) integer pixel coordinates
    """
    if indices is None:
        indices = range(len(landmarks))

    result = []
    for idx in indices:
        if idx < len(landmarks):
            lm = landmarks[idx]
            x = int(lm[0] * width)
            y = int(lm[1] * height)
            result.append((x, y))

    return result
