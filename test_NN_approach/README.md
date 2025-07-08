# Blue Marker Detection System

->> what if I trained a neural network to detect the blue markers? I would be able to manually create datapoints where not all of the markers are clearly visable / the board is shaking (opencv seems to struggle with that). 

->> issue: input is very large (1920 x 1080) (also dependant on the camera res), so i guess i have to work in smaller resolution, implement a convnet, and train it for every new camera. yikes. 

->>how do we make sure the NN always outputs the blue dots in the preassigned order? 


This project provides a complete pipeline for detecting blue markers in images and videos using neural networks. The system consists of four main components that work together:

1. **Data Collection** - Creates a dataset by extracting blue marker positions from a video
2. **Data Augmentation** - Expands the dataset with various transformations
3. **Model Training** - Trains a neural network to detect blue markers
4. **Inference** - Uses the trained model to detect markers in new images/videos

## Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- tqdm
- scikit-learn

Install requirements:

pip install tensorflow opencv-python numpy matplotlib tqdm scikit-learn


## Usage Pipeline

### Step 1: Create a Dataset

Record a video that contains 8 blue markers visible in each frame. Then run:

python detect_blue_markers.py


This script will:
- Prompt for the path to your video
- Process each frame to detect blue markers
- Save frames with exactly 8 detected markers as images
- Create a JSON dataset file with marker positions

### Step 2: Augment the Dataset

Expand your dataset with various transformations:


python augment_dataset.py




This script will:
- Read the original dataset created in Step 1
- Apply various transformations (brightness changes, rotations, shading)
- Create an augmented dataset with more samples

### Step 3: Train the Neural Network

Train a model to detect blue markers:

python train_NN.py


This script will:
- Prompt for training options (model type, image size, etc.)
- Load the augmented dataset
- Train the selected model architecture
- Save the best model and performance metrics

### Step 4: Use the Model for Inference

Use your trained model for inference:

GitHub Copilot
I've reviewed the four Python scripts and confirmed they are compatible with each other. Together they form a complete pipeline for detecting blue markers in images/videos using neural networks. Here's a README file that explains how to use them together:

README.md
README.md+21-1
pip install tensorflow opencv-python numpy matplotlib tqdm scikit-learn


## Usage Pipeline

### Step 1: Create a Dataset

Record a video that contains 8 blue markers visible in each frame. Then run:

python detect_blue_markers.py

for
python augment_dataset.py

python train_NN.py

python inference.py --model models/marker_detector_cnn.h5 --input your_video.mp4 --output result.mp4



Additional options:
- `--size 224`: Set input size to match training (default: 224)
- `--webcam 0`: Use webcam instead of a file
- `--no-display`: Disable display window

## Model Types

The system supports two types of neural network architectures:

1. **CNN** (default): Based on EfficientNetB0 for feature extraction
2. **Transformer**: A Vision Transformer (ViT) inspired architecture

## Example Workflow

```bash
# 1. Create dataset from a video
python detect_blue_markers.py
# Enter: /path/to/your/video.mp4

# 2. Augment the dataset
python augment_dataset.py

# 3. Train neural network
python train_NN.py
# Enter: cnn
# Enter: 224
# Enter: 16
# Enter: 50

# 4. Run inference on a new video
python inference.py --model models/marker_detector_cnn.h5 --input new_video.mp4 --output result.mp4


Tips for Best Results
Use distinct blue markers with good contrast to the background
Ensure consistent lighting when recording the training dataset
Make sure all 8 markers are visible in training frames
For inference, try to maintain similar conditions to the training data