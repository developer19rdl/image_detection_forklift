# Forklift + Person Detection System - Training Guide

## What You Have in This Package

```
forklift-detector-package/
├── models/
│   ├── best_v4.pt                    # YOUR TRAINED MODEL (2 classes: forklift + person)
│   │                                  # Trained on 1080 images, val mAP50 = 88%
│   └── best_v2_single_class.pt        # Older model (1 class: forklift only)
├── src/
│   ├── __init__.py
│   ├── detector.py                    # Core ForkliftDetector class (YOLOv8 wrapper)
│   ├── camera.py                      # CameraManager (webcam/RTSP/video/image)
│   ├── preprocess.py                  # CLAHE, blur, letterbox resize
│   ├── utils.py                       # Drawing, safety zone, FPS counter
│   └── alerts.py                      # Console, CSV, webhook, beep alerts
├── config/
│   └── config.yaml                    # All settings (model, display, alerts, safety zone)
├── results/
│   ├── test_detections/               # 20 detection result images from test set
│   │   ├── detected_image_396_*.jpg    # Green box = forklift, Blue box = person
│   │   └── ...
│   └── training_curves/               # Confusion matrix, PR curves, F1 curve
├── dataset_example/
│   └── data.yaml                      # Example dataset config (2 classes)
├── main.py                            # Real-time camera/video detection
├── main_image.py                      # Batch image detection
├── train.py                           # Training script
├── requirements.txt                   # Python dependencies
└── TRAINING_GUIDE.md                  # THIS FILE
```

---

## Training Results Summary

| Metric | Value |
|--------|-------|
| Dataset | Roboflow forklift (1080 train / 35 val / 35 test) |
| Classes | 2 (forklift, person) |
| Base Model | YOLOv8n (pretrained on COCO) |
| Image Size | 320x320 |
| Batch Size | 2 |
| Epochs Completed | 2 (environment constrained) |
| Val mAP50 | 88.0% |
| Test mAP50 | 68.5% |
| Test Precision | 67.1% |
| Test Recall | 82.9% |
| Model Size | 24 MB |
| Inference Speed | ~40ms/image on CPU |

> **Note**: With only 2 epochs completed due to CPU/memory constraints in the training environment, the model is decent but NOT production-optimized. **Retraining on your machine (especially with GPU) for 50-100 epochs will significantly improve accuracy.**

---

## Part 1: Quick Start (Use the Trained Model)

### 1.1 Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install packages
pip install -r requirements.txt

# If CPU only (lighter, no CUDA needed):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 1.2 Run Detection on Images

```bash
python main_image.py --source path/to/image.jpg --output results/ --show
python main_image.py --source path/to/images_folder/ --output results/ --show
python main_image.py "C:\Users\Administrator\Downloads\Untitled.jpg" --show

```

### 1.3 Run Detection on Webcam

```bash
python main.py --source 0
```

### 1.4 Run Detection on Video

```bash
python main.py --source path/to/video.mp4 --output output_video.avi
```

### 1.5 Python API Usage

```python
from ultralytics import YOLO
import cv2

model = YOLO("models/best_v4.pt")
image = cv2.imread("forklift_scene.jpg")

# Run detection
results = model(image, imgsz=320, conf=0.35)

# Parse results
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        confidence = box.conf[0].cpu().numpy()
        class_id = int(box.cls[0].cpu().numpy())
        class_name = model.names[class_id]  # 'forklift' or 'person'
        print(f"{class_name}: {confidence:.1%} at [{x1},{y1},{x2},{y2}]")

# Draw detections
colors = [(0, 255, 0), (255, 0, 0)]  # green=forklift, blue=person
annotated = image.copy()
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = box.conf[0].cpu().numpy()
        cls = int(box.cls[0].cpu().numpy())
        label = f"{model.names[cls]} {conf:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colors[cls], 2)
        cv2.putText(annotated, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[cls], 2)

cv2.imwrite("output.jpg", annotated)
```

---

## Part 2: How to Train the Model Yourself

### Why Retrain?

The included model was trained for only **2 epochs** due to environment constraints. Retraining on your machine (especially with GPU) for **50-100 epochs** will significantly improve accuracy. You should also retrain if:
- You want to add more classes (e.g., helmet, pallet, warning sign)
- Your forklifts look different from the training data
- You need higher accuracy for production use

### Step 1: Get the Dataset

The dataset used is from **Roboflow**: https://universe.roboflow.com/damla-yaar/forklift-odgis/dataset/1

It contains **1080 training images** with 2 classes:
- **Class 0: forklift** (918 annotations)
- **Class 1: person** (537 annotations)

Download it in YOLOv8 format. You'll get a zip file that extracts to:

```
forklift.v1i.yolov8/
├── data.yaml          # Dataset config
├── train/
│   ├── images/       # 1080 training images
│   └── labels/       # YOLO format .txt labels
├── valid/
│   ├── images/       # 35 validation images
│   └── labels/
└── test/
    ├── images/       # 35 test images
    └── labels/
```

### Step 2: Organize and Configure

1. Extract the zip to a folder like `data/`
2. Edit `data.yaml` to use **absolute paths**:

```yaml
# data.yaml
train: /absolute/path/to/data/train/images
val: /absolute/path/to/data/valid/images
test: /absolute/path/to/data/test/images

nc: 2
names: ['forklift', 'person']
```

### Step 3: Train the Model

**Option A: Using the included train.py**

```bash
# GPU training (recommended)
python train.py --data /path/to/data.yaml --epochs 100 --batch 16 --device 0

# CPU training
python train.py --data /path/to/data.yaml --epochs 100 --batch 8 --device cpu
```

**Option B: Using YOLO CLI directly**

```bash
# GPU
yolo train model=yolov8n.pt data=/path/to/data.yaml epochs=100 batch=16 device=0

# CPU
yolo train model=yolov8n.pt data=/path/to/data.yaml epochs=100 batch=8 device=cpu imgsz=320
```

**Option C: Using Python API (most control)**

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Start from COCO pretrained weights

results = model.train(
    data='/path/to/data.yaml',
    epochs=100,
    batch=16,          # Use 16 for GPU, 4-8 for CPU
    imgsz=640,         # 640 for best accuracy, 320 for speed
    device='0',        # '0' for GPU, 'cpu' for CPU
    patience=20,       # Early stopping if no improvement for 20 epochs
    save=True,
    save_period=10,    # Save checkpoint every 10 epochs
    plots=True,        # Generate training curves
    verbose=True,
)

print(f"mAP50: {results.results_dict['metrics/mAP50(B)']}")
print(f"mAP50-95: {results.results_dict['metrics/mAP50-95(B)']}")
```

### Step 4: Monitor Training

Training prints a table every epoch:

```
Epoch  GPU_mem  box_loss  cls_loss  dfl_loss  Instances  Size
  1/100     1.2G    1.2345    0.6789    1.0123         12    640
```

**Key metrics to watch:**
- `box_loss` — Should decrease (bounding box accuracy improving)
- `cls_loss` — Should decrease (classification accuracy improving)
- `mAP50` — Should increase and plateau (main quality metric)

**After training, check these files:**
- `runs/detect/train/weights/best.pt` — Best model (use this!)
- `runs/detect/train/weights/last.pt` — Last epoch model
- `runs/detect/train/results.png` — Loss and mAP curves
- `runs/detect/train/confusion_matrix.png` — Detection quality matrix

### Step 5: Evaluate on Test Set

```bash
yolo val model=runs/detect/train/weights/best.pt data=/path/to/data.yaml split=test
```

Or in Python:

```python
from ultralytics import YOLO
model = YOLO('runs/detect/train/weights/best.pt')
results = model.val(data='/path/to/data.yaml', split='test')
print(f"Test mAP50: {results.results_dict['metrics/mAP50(B)']}")
```

### Step 6: Use Your Trained Model

```bash
# Copy best model to the models/ folder
cp runs/detect/train/weights/best.pt models/best_v4.pt

# Update config.yaml to point to your new model
# Then run detection as normal
python main.py --source 0
python main_image.py --source test_images/ --output results/
```

---

## Part 3: Training Parameters Reference

| Parameter | Recommended (GPU) | Recommended (CPU) | Description |
|-----------|-------------------|-------------------|-------------|
| `epochs` | 50-100 | 50-100 | Training passes. More = better (with early stopping) |
| `batch` | 16-32 | 4-8 | Images per batch. Reduce if OOM error |
| `imgsz` | 640 | 320-416 | Larger = more accurate but slower |
| `model` | yolov8n.pt | yolov8n.pt | Base model. Try yolov8s.pt for more accuracy |
| `device` | 0 | cpu | GPU device number or 'cpu' |
| `patience` | 20 | 15 | Stop early if no mAP improvement |
| `lr0` | 0.01 | 0.01 | Initial learning rate |
| `augment` | True | True | Built-in augmentation (mosaic, flip, HSV) |

### Hardware Recommendations

| Hardware | Epochs | Batch | Img Size | Expected mAP50 |
|----------|--------|-------|----------|----------------|
| CPU only, 8GB RAM | 100 | 4-8 | 320 | 80-90% |
| CPU only, 16GB RAM | 100 | 8-16 | 416 | 85-92% |
| GPU 4GB VRAM | 100 | 16 | 416-640 | 90-95% |
| GPU 8GB+ VRAM | 100 | 32 | 640 | 92-97% |

---

## Part 4: Creating Your Own Dataset

### Option A: Use Roboflow (Easiest)

1. Go to https://roboflow.com
2. Search "forklift" or upload your own images
3. Annotate online using their tools
4. Export as **YOLOv8** format
5. Download and use directly

### Option B: Annotate Manually with LabelImg

```bash
pip install labelImg
labelImg
```

1. Open your images directory
2. Set save dir for labels
3. Switch format to **YOLO** (left panel)
4. Draw bounding boxes:
   - Class 0 = forklift
   - Class 1 = person (or any other class you want)
5. Save each image (Ctrl+S)

### YOLO Label Format

Each image needs a `.txt` file (same name) with:

```
class_id x_center y_center width height
```

All values normalized 0-1. Example:

```
# forklift bounding box
0 0.45 0.55 0.35 0.70
# person bounding box  
1 0.80 0.40 0.15 0.50
```

### Dataset Split

Organize into 80/10/10 split:

```
my_dataset/
├── data.yaml
├── train/images/  (80%)
├── train/labels/
├── valid/images/  (10%)
├── valid/labels/
├── test/images/   (10%)
└── test/labels/
```

---

## Part 5: Improving Detection Accuracy

### Model detects too few (low recall):
- Lower `conf_threshold` in config.yaml (try 0.15)
- Train for more epochs (50-100+)
- Add more training images of missed scenarios
- Use larger `imgsz` (416 or 640)
- Try bigger model: `yolov8s.pt` or `yolov8m.pt`

### Model detects non-forklifts (low precision):
- Raise `conf_threshold` (try 0.5)
- Add negative images (warehouse scenes WITHOUT forklifts, no label files needed)
- Check annotations: ensure boxes are tight around forklifts only
- Train for more epochs

### For production deployment:
- Export to ONNX: `model.export(format='onnx')`
- Export to TensorRT: `model.export(format='engine')` (GPU only)
- Use OpenCV DNN module for inference without PyTorch dependency

---

## Part 6: Adding More Classes

To detect additional objects (helmets, pallets, safety signs):

1. Add class names to `data.yaml`:

```yaml
nc: 4
names: ['forklift', 'person', 'helmet', 'pallet']
```

2. Annotate new images with class IDs 2 and 3
3. Retrain the model
4. Update config.yaml `class_names` list

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Reduce `--batch` to 4 or 2 |
| `ModuleNotFoundError: ultralytics` | `pip install ultralytics` |
| Model not detecting | Check `config.yaml` weights path |
| Low FPS on CPU | Reduce `imgsz` to 320, use `yolov8n.pt` |
| Poor accuracy | Train 50+ epochs, add more images |
| `FileNotFoundError: best.pt` | Run training first to generate model |
| Training too slow on CPU | Use `imgsz=320`, `batch=4`, `workers=0` |
| Wrong class names | Update `data.yaml` names list and retrain |
