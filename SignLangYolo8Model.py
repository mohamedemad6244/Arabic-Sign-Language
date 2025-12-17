# ===============================
# 1. Imports
# ===============================
import os
from pathlib import Path
from ultralytics import YOLO
import shutil
import kagglehub

# ===============================
# 2. Download Dataset
# ===============================
path = kagglehub.dataset_download("ammarsayedtaha/arabic-sign-language-dataset-2022")
print("Dataset path:", path)

DATASET_PATH = os.path.join(path, "datasets")
YOLO_PATH = "/content/yolo_dataset"

# ===============================
# 3. Prepare YOLO Dataset
# ===============================
os.makedirs(YOLO_PATH, exist_ok=True)

class_names = []

for split in ["train", "valid"]:
    split_path = Path(DATASET_PATH) / split
    images_out = Path(YOLO_PATH) / split / "images"
    labels_out = Path(YOLO_PATH) / split / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    
    for class_idx, class_dir in enumerate(sorted(split_path.iterdir())):
        if not class_dir.is_dir():
            continue
        
        # Save class names for YAML
        if split == "train":
            class_names.append(class_dir.name)
        
        for img_path in class_dir.glob("*.jpg"):
            # Copy image
            dest_img_path = images_out / img_path.name
            shutil.copy(img_path, dest_img_path)
            
            # Create YOLO label (full image bounding box)
            label_path = labels_out / f"{img_path.stem}.txt"
            with open(label_path, "w") as f:
                f.write(f"{class_idx} 0.5 0.5 1.0 1.0\n")

print(f"Number of classes: {len(class_names)}")
print(f"Classes: {class_names}")

# ===============================
# 4. Create YOLOv8 YAML Config
# ===============================
yaml_path = "/content/yolo_dataset.yaml"
with open(yaml_path, "w") as f:
    f.write(f"train: {YOLO_PATH}/train/images\n")
    f.write(f"val: {YOLO_PATH}/valid/images\n")
    f.write(f"nc: {len(class_names)}\n")
    f.write(f"names: {class_names}\n")

print(f"YOLO YAML config created at {yaml_path}")

# ===============================
# 5. Train YOLOv8 with EarlyStopping
# ===============================
model = YOLO("yolov8n.pt")  # YOLOv8 Nano pre-trained

# Train
model.train(
    data=yaml_path,
    epochs=50,
    imgsz=224,
    batch=16,          # adjust depending on GPU
    name="arabic_sign_yolov8",
    device=0,          # GPU
    patience=10        # EarlyStopping equivalent: stops if no val improvement for 10 epochs
)

# ===============================
# 6. Evaluate on Validation Set
# ===============================
metrics = model.val()
print(metrics)

# ===============================
# 7. Predict on a Sample Image
# ===============================
sample_image = list(Path(YOLO_PATH, "valid", "images").glob("*.jpg"))[0]
results = model.predict(str(sample_image))

# Show the prediction
results[0].show()  # fix: show the first result from the list

# ===============================
# 8. Export YOLOv8 Model to TFLite
# ===============================
model.export(format="tflite")
print("TFLite model saved.")