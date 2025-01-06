import pandas as pd
import os

# Define paths
data_file = './Train.csv'  # Path to your train.csv file
output_labels_folder = './labels/Train/'  # Folder to save YOLO annotations

# Create output folder if it doesn't exist
os.makedirs(output_labels_folder, exist_ok=True)

# Read the train.csv file
print(f"Reading data from: {data_file}")
annotations = pd.read_csv(data_file)

# Process each row in the CSV file
for index, row in annotations.iterrows():
    # Image details
    image_width = row['Width']  # Width of the image
    image_height = row['Height']  # Height of the image
    image_path = row['Path']  # Path to the image file

    # Bounding box details
    x_min = row['Roi.X1']
    y_min = row['Roi.Y1']
    x_max = row['Roi.X2']
    y_max = row['Roi.Y2']

    # Class ID
    class_id = row['ClassId']

    # Calculate bounding box center and size (normalized)
    x_center = (x_min + x_max) / 2 / image_width
    y_center = (y_min + y_max) / 2 / image_height
    bbox_width = (x_max - x_min) / image_width
    bbox_height = (y_max - y_min) / image_height

    # Format YOLO annotation line
    yolo_annotation = f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n"

    # Generate output label file name
    label_filename = os.path.basename(image_path).replace('.jpg', '.txt').replace('.png', '.txt')
    label_path = os.path.join(output_labels_folder, label_filename)

    # Write YOLO annotation to the label file
    with open(label_path, 'w') as label_file:
        label_file.write(yolo_annotation)

    print(f"Processed: {image_path} -> {label_path}")

print("Conversion to YOLO format completed.")
