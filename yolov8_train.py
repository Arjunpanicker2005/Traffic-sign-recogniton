from ultralytics import YOLO

# Paths
MODEL_NAME = "yolov8n.pt"  # Replace with yolov8s.pt, yolov8m.pt, etc., for larger models
DATA_YAML = "data.yaml"  # Path to your data.yaml file
EPOCHS = 1 # Number of training epochs
IMG_SIZE = 640  # Image size for training
BATCH_SIZE = 16  # Batch size for training

# 1. Train the YOLOv8 Model
print("Starting training...")
model = YOLO(MODEL_NAME)  # Load pre-trained YOLOv8 model
model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE
)

# 2. Validate the Model
print("\nValidating the model...")
model.val()

# 3. Predict on a Test Image
TEST_IMAGE = "C:/Users/Ajith Kumar KP/Documents/yolo/dataset/val/images/00000.png"  # Path to your test image
print(f"\nRunning prediction on: {TEST_IMAGE}")
results = model.predict(source=TEST_IMAGE, save=True, imgsz=IMG_SIZE)
print(f"Predictions saved to: {results[0].save_dir}")

