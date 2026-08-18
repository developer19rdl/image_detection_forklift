import cv2
import numpy as np
import time
import logging
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class ForkliftDetector:
    """Core YOLOv8-based forklift detector with OpenCV integration."""

    def __init__(self, weights: str, conf_threshold: float = 0.35,
                 iou_threshold: float = 0.45, imgsz: int = 640,
                 device: str = "cpu", max_detections: int = 20):
        self.weights = weights
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device
        self.max_detections = max_detections

        self.model = None
        self.class_names = []
        self.inference_times: List[float] = []

        self._load_model()

    def _load_model(self) -> None:
        """Load the YOLOv8 model."""
        logger.info(f"Loading model from: {self.weights}")
        logger.info(f"Device: {self.device} | Image size: {self.imgsz}")
        t0 = time.time()
        self.model = YOLO(self.weights)
        self.class_names = list(self.model.names.values())
        elapsed = time.time() - t0
        logger.info(f"Model loaded in {elapsed:.2f}s — Classes: {self.class_names}")

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run detection on a single BGR frame.
        Returns list of detection dicts with keys:
          bbox (x1,y1,x2,y2), confidence, class_id, class_name, center
        """
        t0 = time.time()
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            max_det=self.max_detections,
        )
        inference_time = time.time() - t0
        self.inference_times.append(inference_time)
        # Keep only last 100 measurements for rolling average
        if len(self.inference_times) > 100:
            self.inference_times = self.inference_times[-100:]

        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": round(conf, 4),
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "center": (cx, cy),
                })
        return detections

    def get_avg_inference_time(self) -> float:
        """Return rolling average inference time in milliseconds."""
        if not self.inference_times:
            return 0.0
        return (sum(self.inference_times) / len(self.inference_times)) * 1000
