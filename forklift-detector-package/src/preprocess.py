import cv2
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class Preprocessor:
    """Image preprocessing pipeline for detection input."""

    def __init__(self, target_size: Tuple[int, int] = (640, 640),
                 clahe_enabled: bool = False,
                 blur_kernel: int = 0):
        self.target_size = target_size
        self.clahe_enabled = clahe_enabled
        self.blur_kernel = blur_kernel
        self._clahe = None
        if clahe_enabled:
            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Apply the full preprocessing pipeline to a BGR frame."""
        result = frame.copy()

        # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if self.clahe_enabled and self._clahe is not None:
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 2. Gaussian blur for noise reduction
        if self.blur_kernel > 0:
            k = self.blur_kernel if self.blur_kernel % 2 == 1 else self.blur_kernel + 1
            result = cv2.GaussianBlur(result, (k, k), 0)

        # 3. Resize
        if self.target_size != (0, 0):
            result = cv2.resize(result, self.target_size)

        return result

    @staticmethod
    def letterbox(frame: np.ndarray, target_size: int = 640,
                  color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Resize and pad image to target_size maintaining aspect ratio.
        Returns: (letterboxed_image, scale_ratio, (pad_left, pad_top))
        """
        h, w = frame.shape[:2]
        scale = min(target_size / w, target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        pad_left = (target_size - new_w) // 2
        pad_top = (target_size - new_h) // 2
        padded = cv2.copyMakeBorder(
            resized, pad_top, target_size - new_h - pad_top,
            pad_left, target_size - new_w - pad_left,
            cv2.BORDER_CONSTANT, value=color
        )
        return padded, scale, (pad_left, pad_top)
