import cv2
import time
import logging
from typing import Optional, Generator, Tuple

logger = logging.getLogger(__name__)


class CameraManager:
    """Manages camera, video file, RTSP stream, or image folder input."""

    def __init__(self, source, resize_width: int = 0, resize_height: int = 0,
                 frame_skip: int = 0):
        self.source = source
        self.resize_w = resize_width
        self.resize_h = resize_height
        self.frame_skip = frame_skip
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        self.fps = 0.0
        self._is_image_source = False
        self._image_list: list = []
        self._image_index = 0

    def open(self) -> bool:
        """Open the video source. Returns True on success."""
        source_str = str(self.source)

        # Check if source is an image file
        if source_str.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            self._is_image_source = True
            img = cv2.imread(source_str)
            if img is None:
                logger.error(f"Cannot read image: {source_str}")
                return False
            self._image_list = [img]
            self.fps = 1.0
            logger.info(f"Loaded single image: {source_str} ({img.shape[1]}x{img.shape[0]})")
            return True

        # Check if source is a directory of images
        import os
        if os.path.isdir(source_str):
            self._is_image_source = True
            exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            self._image_list = []
            for fname in sorted(os.listdir(source_str)):
                if fname.lower().endswith(exts):
                    img = cv2.imread(os.path.join(source_str, fname))
                    if img is not None:
                        self._image_list.append(img)
            if not self._image_list:
                logger.error(f"No valid images found in: {source_str}")
                return False
            self.fps = 1.0
            logger.info(f"Loaded {len(self._image_list)} images from: {source_str}")
            return True

        # Video source (webcam, RTSP, video file)
        self._is_image_source = False
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logger.error(f"Cannot open video source: {self.source}")
            return False

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Opened video source: {self.source} ({width}x{height} @ {self.fps:.1f} FPS)")
        return True

    def read_frame(self) -> Tuple[bool, Optional[object]]:
        """Read next frame. Returns (success, frame_or_None)."""
        if self._is_image_source:
            if self._image_index >= len(self._image_list):
                return False, None
            frame = self._image_list[self._image_index].copy()
            self._image_index += 1
            self.frame_count += 1
            return True, frame

        if self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            return False, None

        # Frame skipping for performance
        if self.frame_skip > 0 and self.frame_count % (self.frame_skip + 1) != 0:
            self.frame_count += 1
            return self.read_frame()  # Recursive skip

        self.frame_count += 1
        return True, frame

    def resize_frame(self, frame) -> object:
        """Resize frame if resize dimensions are set."""
        if self.resize_w > 0 and self.resize_h > 0:
            return cv2.resize(frame, (self.resize_w, self.resize_h))
        return frame

    def release(self) -> None:
        """Release resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        logger.info("Camera released.")

    def __del__(self):
        self.release()
