# Traffic-sign-recogniton


This project implements a traffic sign detection and classification system using the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The system is designed to help in autonomous driving and driver assistance by accurately recognizing traffic signs.

## Features
- Preprocessing of images (resizing, normalization, and augmentation)
- Training a YOLOv8 model for sign detection and classification
- Evaluation of model performance using accuracy and loss metrics
  

## Dataset
The [GTSRB dataset]contains over 50,000 images of traffic signs belonging to 43 classes. Each image is labeled with the corresponding traffic sign type.

## Project Structure
```
traffic_sign_detection/
├── data/                 # Dataset directory
├── models/               # Saved models
├── scripts/              # Training, evaluation, and detection scripts
├── requirements.txt      # Python dependencies
├── README.md             # Project description
└── main.py               # Main script to run the detection system
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/traffic_sign_detection.git
   cd traffic_sign_detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the GTSRB dataset and place it in the `data/` directory.

## Usage

### Training
To train the model, run the training script:
```bash
python scripts/train.py --epochs 20 --batch-size 32 --learning-rate 0.001
```

### Evaluation
To evaluate the model, run the evaluation script:
```bash
python scripts/evaluate.py --model models/traffic_sign_model.h5
```

### Image Detection
To perform real-time detection on a video stream:
```bash
python scripts/detect.py --video input_video.mp4
```


## Contributing
Contributions are welcome! Please fork this repository and submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## References
- [GTSRB Download](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- [YOLOv8 Documentation](https://docs.ultralytics.com)
  
