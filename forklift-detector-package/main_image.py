import sys
import os
import cv2
import yaml
import argparse
import glob

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.detector import ForkliftDetector
from src.alerts import AlertSystem
from src.utils import DrawingUtils


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Forklift Detection System - Image Mode")
    parser.add_argument("images", nargs="+", help="Image file path(s) or directory")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--output", type=str, default=None,
                        help="Directory to save detection results")
    parser.add_argument("--show", action="store_true",
                        help="Display results in a window")
    parser.add_argument("--wait-key", type=int, default=0,
                        help="Wait key delay in ms (0 = wait forever)")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    model_cfg = config["model"]
    alert_cfg = config["alerts"]
    display_cfg = config["display"]

    # Collect image paths
    image_paths = []
    for path in args.images:
        if os.path.isdir(path):
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                image_paths.extend(glob.glob(os.path.join(path, ext)))
        elif os.path.isfile(path):
            image_paths.append(path)
        else:
            print(f"Warning: {path} not found, skipping.")

    if not image_paths:
        print("No valid images found. Exiting.")
        sys.exit(1)

    # Initialize
    print("=" * 60)
    print("  FORKLIFT DETECTION SYSTEM - Image Mode")
    print("=" * 60)
    print(f"  Images to process: {len(image_paths)}")

    alert_sys = AlertSystem(
        log_console=alert_cfg["log_console"],
        log_level=alert_cfg["log_level"],
        log_csv=alert_cfg["log_csv"],
        csv_path=alert_cfg["csv_path"],
    )

    detector = ForkliftDetector(
        weights=model_cfg["weights"],
        conf_threshold=model_cfg["conf_threshold"],
        iou_threshold=model_cfg["iou_threshold"],
        imgsz=model_cfg["imgsz"],
        device=model_cfg["device"],
        max_detections=model_cfg["max_detections"],
    )

    # Process all images
    total_detections = 0
    for i, img_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] Processing: {img_path}")
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  [SKIP] Cannot read image.")
            continue

        detections = detector.detect(frame)
        total_detections += len(detections)
        alert_sys.log_detections(detections, i)

        if display_cfg["show_overlay"]:
            frame = DrawingUtils.draw_detections(
                frame, detections,
                box_color=tuple(display_cfg["box_color"]),
                box_thickness=display_cfg["box_thickness"],
                font_scale=display_cfg["font_scale"],
                show_confidence=display_cfg["show_confidence"],
            )
        frame = DrawingUtils.draw_detection_count(frame, len(detections))

        if args.output:
            os.makedirs(args.output, exist_ok=True)
            out_path = os.path.join(args.output, f"detected_{os.path.basename(img_path)}")
            cv2.imwrite(out_path, frame)
            print(f"  [SAVED] {out_path}")

        if args.show:
            cv2.imshow("Forklift Detection - Image Mode", frame)
            key = cv2.waitKey(args.wait_key) & 0xFF
            if key == ord('q'):
                break

    cv2.destroyAllWindows()

    print(f"\n{'=' * 60}")
    print(f"  Summary")
    print(f"{'=' * 60}")
    print(f"  Images processed: {len(image_paths)}")
    print(f"  Total detections: {total_detections}")
    print(f"  Avg inference: {detector.get_avg_inference_time():.1f} ms")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
