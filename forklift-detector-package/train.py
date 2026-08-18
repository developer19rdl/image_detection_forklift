import sys
import os
import argparse
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 on custom forklift dataset"
    )
    parser.add_argument(
        "--data", type=str, default="data/data.yaml",
        help="Path to dataset YAML config"
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help="Base model (yolov8n.pt, yolov8s.pt, yolov8m.pt)"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu, 0 (gpu), mps")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument(
        "--name", type=str, default="forklift_train",
        help="Experiment name (saved in runs/detect/<name>)"
    )
    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="Resume from a pretrained .pt checkpoint"
    )
    parser.add_argument(
        "--augment", action="store_true", default=True,
        help="Enable data augmentation"
    )
    args = parser.parse_args()

    # Validate dataset config exists
    if not os.path.isfile(args.data):
        print(f"ERROR: Dataset config not found: {args.data}")
        print("\nBefore training, you must:")
        print("  1. Collect forklift images and place them in data/images/")
        print("  2. Annotate them (use Roboflow or LabelImg) in YOLO format")
        print("  3. Organize into data/images/train/ and data/images/val/")
        print("  4. Update data/data.yaml with correct paths")
        print("\nSee README.md for detailed instructions.")
        sys.exit(1)

    print("=" * 60)
    print("  FORKLIFT DETECTION - MODEL TRAINING")
    print("=" * 60)
    print(f"  Base model:   {args.model}")
    print(f"  Dataset:      {args.data}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Image size:   {args.imgsz}")
    print(f"  Batch size:   {args.batch}")
    print(f"  Device:       {args.device}")
    print(f"  Augmentation: {args.augment}")
    print("=" * 60)

    # Load model
    weights = args.pretrained if args.pretrained else args.model
    model = YOLO(weights)

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        name=args.name,
        augment=args.augment,
        project="runs/detect",
        exist_ok=True,
        verbose=True,
    )

    # Export to ONNX for deployment
    print("\n" + "=" * 60)
    print("  EXPORTING MODEL")
    print("=" * 60)
    best_pt = f"runs/detect/{args.name}/weights/best.pt"
    if os.path.isfile(best_pt):
        export_model = YOLO(best_pt)
        export_model.export(format="onnx", imgsz=args.imgsz)
        print(f"  Model exported to: {best_pt.replace('.pt', '.onnx')}")

        # Copy best model to models/ directory
        os.makedirs("models", exist_ok=True)
        import shutil
        shutil.copy2(best_pt, "models/best.pt")
        print(f"  Best model copied to: models/best.pt")

        # Update config to use trained model
        config_path = "config/config.yaml"
        if os.path.isfile(config_path):
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            cfg["model"]["weights"] = "models/best.pt"
            with open(config_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)
            print(f"  Config updated to use trained model.")

    print(f"\n{'=' * 60}")
    print(f"  Training complete!")
    print(f"  Results saved to: runs/detect/{args.name}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
