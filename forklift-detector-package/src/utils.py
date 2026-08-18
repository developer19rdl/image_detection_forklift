import cv2
import numpy as np
import time
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class DrawingUtils:
    """Utility class for drawing detection overlays on frames."""

    # Color palette for different classes (BGR format)
    COLORS = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (128, 0, 255),  # Purple
        (0, 128, 255),  # Orange
    ]

    @classmethod
    def draw_detections(cls, frame: np.ndarray, detections: List[Dict],
                        box_color: Tuple[int, int, int] = None,
                        box_thickness: int = 2,
                        font_scale: float = 0.7,
                        show_confidence: bool = True) -> np.ndarray:
        """Draw bounding boxes and labels on the frame."""
        overlay = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            cls_name = det["class_name"]

            color = box_color if box_color else cls.COLORS[det["class_id"] % len(cls.COLORS)]

            # Draw bounding box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, box_thickness)

            # Build label text
            label = cls_name
            if show_confidence:
                label += f" {conf:.0%}"

            # Calculate label background size
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            label_y = max(y1 - 10, th + 10)

            # Draw label background
            cv2.rectangle(overlay, (x1, label_y - th - 5), (x1 + tw + 4, label_y + 2), color, -1)

            # Draw label text
            cv2.putText(overlay, label, (x1 + 2, label_y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

            # Draw center point
            cx, cy = det["center"]
            cv2.circle(overlay, (cx, cy), 4, color, -1)

        return overlay

    @classmethod
    def draw_safety_zone(cls, frame: np.ndarray, points: List[List[int]],
                         color: Tuple[int, int, int] = (0, 255, 255),
                         thickness: int = 2) -> np.ndarray:
        """Draw a polygon safety zone on the frame."""
        overlay = frame.copy()
        if len(points) >= 3:
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(overlay, [pts], True, color, thickness)
            # Semi-transparent fill
            fill = overlay.copy()
            cv2.fillPoly(fill, [pts], color)
            cv2.addWeighted(fill, 0.15, overlay, 0.85, 0, overlay)
        return overlay

    @classmethod
    def draw_fps(cls, frame: np.ndarray, fps: float, inference_ms: float,
                 position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        """Draw FPS and inference time overlay."""
        text = f"FPS: {fps:.1f} | Inference: {inference_ms:.1f}ms"
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2, cv2.LINE_AA)
        return frame

    @classmethod
    def draw_detection_count(cls, frame: np.ndarray, count: int,
                             position: Tuple[int, int] = (10, 60)) -> np.ndarray:
        """Draw forklift detection count on frame."""
        color = (0, 0, 255) if count > 0 else (0, 255, 0)
        text = f"Forklifts: {count}"
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color, 2, cv2.LINE_AA)
        return frame


class PointInPolygon:
    """Check if a point is inside a polygon (safety zone)."""

    @staticmethod
    def is_inside(point: Tuple[int, int], polygon: List[List[int]]) -> bool:
        """Ray-casting algorithm to check if point is inside polygon."""
        x, y = point
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside


class FPSCounter:
    """Simple FPS counter."""

    def __init__(self):
        self._start_time = time.time()
        self._frame_count = 0
        self._fps = 0.0

    def tick(self) -> float:
        """Call once per frame. Returns current FPS."""
        self._frame_count += 1
        elapsed = time.time() - self._start_time
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._start_time = time.time()
        return self._fps

    def reset(self):
        self._start_time = time.time()
        self._frame_count = 0
        self._fps = 0.0
