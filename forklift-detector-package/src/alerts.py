import csv
import os
import time
import logging
import requests
from datetime import datetime
from typing import List, Dict

logger = logging.getLogger(__name__)


class AlertSystem:
    """Handles logging, sound alerts, CSV recording, and webhook notifications."""

    def __init__(self, log_console: bool = True, log_level: str = "INFO",
                    log_csv: bool = False, csv_path: str = "detections.csv",
                    sound_alert: bool = False,
                    webhook_url: str = "",
                    webhook_cooldown: int = 30):
        self.log_console = log_console
        self.log_csv = log_csv
        self.csv_path = csv_path
        self.sound_alert = sound_alert
        self.webhook_url = webhook_url
        self.webhook_cooldown = webhook_cooldown
        self._last_webhook_time = 0
        self._csv_initialized = False

        # Configure root logging
        level = getattr(logging, log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    def _init_csv(self):
        """Initialize CSV file with headers if not exists."""
        if not self._csv_initialized and self.log_csv:
            file_exists = os.path.isfile(self.csv_path)
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        "timestamp", "class_name", "confidence",
                        "x1", "y1", "x2", "y2", "center_x", "center_y"
                    ])
            self._csv_initialized = True

    def log_detections(self, detections: List[Dict], frame_num: int = 0) -> None:
        """Log detections to console and CSV."""
        if not detections:
            return
        self._init_csv()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = det["center"]
            msg = (
                f"[Frame {frame_num}] {det['class_name']} detected | "
                f"Conf: {det['confidence']:.2%} | "
                f"BBox: ({x1},{y1})-({x2},{y2}) | Center: ({cx},{cy})"
            )
            if self.log_console:
                logger.info(msg)
            if self.log_csv:
                try:
                    with open(self.csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            timestamp, det["class_name"], det["confidence"],
                            x1, y1, x2, y2, cx, cy
                        ])
                except Exception as e:
                    logger.error(f"CSV write error: {e}")

    def send_webhook(self, detections: List[Dict]) -> bool:
        """Send detection alert to webhook URL with cooldown."""
        if not self.webhook_url or not detections:
            return False
        now = time.time()
        if now - self._last_webhook_time < self.webhook_cooldown:
            return False
        self._last_webhook_time = now

        payload = {
            "text": f"Forklift Detected! {len(detections)} forklift(s) found.",
            "timestamp": datetime.now().isoformat(),
            "detections": [
                {
                    "class": d["class_name"],
                    "confidence": d["confidence"],
                    "center": list(d["center"]),
                }
                for d in detections
            ],
        }
        try:
            resp = requests.post(self.webhook_url, json=payload, timeout=5)
            logger.info(f"Webhook sent: {resp.status_code}")
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"Webhook failed: {e}")
            return False

    def play_beep(self) -> None:
        """Play a short beep sound (cross-platform)."""
        if not self.sound_alert:
            return
        try:
            import winsound
            winsound.Beep(1000, 200)
        except ImportError:
            try:
                import subprocess
                subprocess.run(["printf", "\a"], capture_output=True)
            except Exception:
                pass
