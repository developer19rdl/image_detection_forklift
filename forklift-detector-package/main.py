import sys
import os
import cv2
import yaml
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.detector import ForkliftDetector
from src.camera import CameraManager
from src.alerts import AlertSystem
from src.utils import DrawingUtils, FPSCounter, PointInPolygon


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Forklift Detection System - Camera/Video Mode")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--source", type=str, default=None,
                        help="Override input source (0=webcam, rtsp://..., path/to/video.mp4)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save output video to file (e.g., output.avi)")
    parser.add_argument("--no-display", action="store_true",
                        help="Run headless (no window display)")
    parser.add_argument("--save-frames", type=str, default=None,
                        help="Save detection frames to directory")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    model_cfg = config["model"]
    input_cfg = config["input"]
    safety_cfg = config["safety_zone"]
    alert_cfg = config["alerts"]
    display_cfg = config["display"]

    # Override source if provided via CLI
    source = args.source if args.source is not None else input_cfg["source"]

    # Initialize components
    print("=" * 60)
    print("  FORKLIFT DETECTION SYSTEM")
    print("=" * 60)

    alert_sys = AlertSystem(
        log_console=alert_cfg["log_console"],
        log_level=alert_cfg["log_level"],
        log_csv=alert_cfg["log_csv"],
        csv_path=alert_cfg["csv_path"],
        sound_alert=alert_cfg["sound_alert"],
        webhook_url=alert_cfg.get("webhook_url", ""),
        webhook_cooldown=alert_cfg.get("webhook_cooldown", 30),
    )

    detector = ForkliftDetector(
        weights=model_cfg["weights"],
        conf_threshold=model_cfg["conf_threshold"],
        iou_threshold=model_cfg["iou_threshold"],
        imgsz=model_cfg["imgsz"],
        device=model_cfg["device"],
        max_detections=model_cfg["max_detections"],
    )

    camera = CameraManager(
        source=source,
        resize_width=input_cfg.get("resize_width", 0),
        resize_height=input_cfg.get("resize_height", 0),
        frame_skip=input_cfg.get("frame_skip", 0),
    )

    if not camera.open():
        print(f"ERROR: Could not open source: {source}")
        print("Make sure your camera is connected or the file/path exists.")
        sys.exit(1)

    fps_counter = FPSCounter()
    safety_points = safety_cfg.get("points", []) if safety_cfg.get("enabled", False) else []

    # Video writer for saving output
    video_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        # Will initialize after first frame

    # Frame save directory
    if args.save_frames:
        os.makedirs(args.save_frames, exist_ok=True)
        saved_count = 0

    print(f"\nSource: {source}")
    print(f"Model:  {model_cfg['weights']}")
    print(f"Device: {model_cfg['device']}")
    print(f"Press 'q' to quit | 's' to save frame | 'p' to pause\n")

    paused = False
    frame_num = 0

    try:
        while True:
            if paused:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('p'):
                    paused = False
                elif key == ord('q'):
                    break
                continue

            ret, frame = camera.read_frame()
            if not ret:
                print("\nStream ended or no more frames.")
                break

            frame = camera.resize_frame(frame)
            frame_num += 1

            # Run detection
            detections = detector.detect(frame)

            # Safety zone check
            zone_violations = []
            if safety_points:
                for det in detections:
                    inside = PointInPolygon.is_inside(det["center"], safety_points)
                    mode = safety_cfg.get("mode", "inside")
                    if (mode == "inside" and inside) or (mode == "outside" and not inside):
                        zone_violations.append(det)

            # Draw overlays
            if display_cfg["show_overlay"]:
                frame = DrawingUtils.draw_detections(
                    frame, detections,
                    box_color=tuple(display_cfg["box_color"]),
                    box_thickness=display_cfg["box_thickness"],
                    font_scale=display_cfg["font_scale"],
                    show_confidence=display_cfg["show_confidence"],
                )
                if safety_points:
                    frame = DrawingUtils.draw_safety_zone(
                        frame, safety_points,
                        color=tuple(safety_cfg.get("color", [0, 255, 255]))
                    )
                if display_cfg["show_fps"]:
                    fps = fps_counter.tick()
                    frame = DrawingUtils.draw_fps(frame, fps, detector.get_avg_inference_time())

            frame = DrawingUtils.draw_detection_count(frame, len(detections))

            # Log and alert
            alert_sys.log_detections(detections, frame_num)
            if zone_violations:
                alert_sys.log_detections(zone_violations, frame_num)
                alert_sys.play_beep()
                alert_sys.send_webhook(zone_violations)

            # Save detection frames
            if args.save_frames and detections:
                saved_count += 1
                cv2.imwrite(os.path.join(args.save_frames, f"detect_{frame_num:06d}.jpg"), frame)

            # Write output video
            if args.output and video_writer is None:
                h, w = frame.shape[:2]
                video_writer = cv2.VideoWriter(args.output, fourcc, 20.0, (w, h))
            if video_writer:
                video_writer.write(frame)

            # Display
            if not args.no_display:
                cv2.imshow(display_cfg["window_name"], frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite(f"snapshot_{frame_num}.jpg", frame)
                    print(f"Snapshot saved: snapshot_{frame_num}.jpg")
                elif key == ord('p'):
                    paused = True

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        camera.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        print(f"\n{'=' * 60}")
        print(f"  Session Summary")
        print(f"{'=' * 60}")
        print(f"  Total frames processed: {frame_num}")
        print(f"  Avg inference time: {detector.get_avg_inference_time():.1f} ms")
        if args.save_frames:
            print(f"  Detection frames saved: {saved_count}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
