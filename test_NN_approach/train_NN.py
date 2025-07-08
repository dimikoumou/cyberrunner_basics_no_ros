import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.applications import EfficientNetB0
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import platform

# Improved GPU configuration
def configure_gpu():
    """Configure GPU based on available hardware (Mac GPU or CUDA)"""
    is_mac = platform.system() == 'Darwin'
    gpu_available = False
    gpu_name = "None"
    
    if is_mac:
        # Check for Mac GPU (Metal)
        try:
            # Configure TensorFlow to use Metal
            physical_devices = tf.config.list_physical_devices('GPU')
            if len(physical_devices) > 0:
                tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                gpu_available = True
                gpu_name = "Mac GPU (Metal)"
                print(f"Using {gpu_name}")
            else:
                print("No Mac GPU detected. Using CPU.")
        except Exception as e:
            print(f"Error configuring Mac GPU: {e}")
            print("Falling back to CPU.")
    else:
        # For other platforms (CUDA)
        try:
            physical_devices = tf.config.list_physical_devices('GPU')
            if len(physical_devices) > 0:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                gpu_available = True
                gpu_name = f"CUDA GPU ({len(physical_devices)} detected)"
                print(f"Using {gpu_name}")
                
                # Set TensorFlow performance optimizations
                tf.config.optimizer.set_jit(True)  # Enable XLA
            else:
                print("No CUDA GPU detected. Using CPU.")
        except Exception as e:
            print(f"Error configuring CUDA GPU: {e}")
            print("Falling back to CPU.")
    
    # Set additional performance configurations
    if gpu_available:
        try:
            # Set additional performance configurations
            tf.config.threading.set_inter_op_parallelism_threads(4)
            tf.config.threading.set_intra_op_parallelism_threads(4)
        except Exception as e:
            print(f"Warning: Could not set additional optimizations: {e}")
    
    return gpu_available, gpu_name

# Configure GPU at module level
GPU_AVAILABLE, GPU_NAME = configure_gpu()

class BlueMarkerDetector:
    def __init__(self, batch_size=16, epochs=50):
        self.model_type = 'cnn'  # Always use CNN
        self.input_shape = None  # Will be auto-detected
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                          f"models/marker_detector_{self.model_type}.h5")
        self.gpu_available = GPU_AVAILABLE
        self.gpu_name = GPU_NAME
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
    
    def build_cnn_model(self):
        """Build a CNN model for marker detection"""
        # Use mixed precision on GPU for better performance
        if self.gpu_available:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print(f"Using mixed precision policy: {policy.name}")
        
        base_model = EfficientNetB0(include_top=False, 
                                    weights='imagenet', 
                                    input_shape=self.input_shape)
        
        # Freeze the base model layers
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16)  # 8 markers x 2 coordinates (x,y)
        ])
        
        return model
    
    def detect_image_size(self, dataset_path):
        """Detect the image size from the first valid image in the dataset"""
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        images_dir = os.path.join(os.path.dirname(dataset_path), "images")
        
        # Try to find a valid image to determine size
        for item in dataset:
            img_path = os.path.join(images_dir, item["image"])
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                # Use the size or adjust to ensure it's divisible by 32 (for EfficientNet)
                size = max(h, w)
                # Round to nearest multiple of 32
                size = ((size + 31) // 32) * 32
                print(f"Auto-detected input size: {size}x{size}")
                return (size, size, 3)
        
        # Default size if no valid images are found
        print("Could not detect input size, using default: 224x224")
        return (224, 224, 3)
    
    def load_dataset(self, dataset_path):
        """Load the augmented dataset for training"""
        # Auto-detect image size if not already set
        if self.input_shape is None:
            self.input_shape = self.detect_image_size(dataset_path)
            
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        images_dir = os.path.join(os.path.dirname(dataset_path), "images")
        
        X = []
        y = []
        
        print("Loading dataset...")
        for item in tqdm(dataset):
            img_path = os.path.join(images_dir, item["image"])
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Resize image
            img_resized = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
            # Convert BGR to RGB (for tensorflow)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            # Normalize image to [0, 1]
            img_normalized = img_rgb / 255.0
            
            # Get original image dimensions for normalization
            orig_h, orig_w = img.shape[:2]
            
            # Flatten the centers list and normalize to [0, 1]
            centers_flat = []
            for center in item["centers"]:
                norm_x = center[0] / orig_w
                norm_y = center[1] / orig_h
                centers_flat.extend([norm_x, norm_y])
            
            X.append(img_normalized)
            y.append(centers_flat)
        
        return np.array(X), np.array(y)
    
    def train(self, dataset_path):
        """Train the neural network on the augmented dataset"""
        # Load dataset
        X, y = self.load_dataset(dataset_path)
        print(f"Dataset loaded: {X.shape[0]} samples")
        
        # Split into train, validation, test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Build model based on type
        self.model = self.build_cnn_model()
        
        # Compile the model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"]
        )
        
        # Display model summary
        self.model.summary()
        
        # Define callbacks
        callbacks_list = [
            callbacks.ModelCheckpoint(
                self.checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                verbose=1,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks_list
        )
        
        # Evaluate on test set
        test_loss, test_mae = self.model.evaluate(X_test, y_test)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        
        # Plot training history
        self.plot_training_history(history)
        
        # Visualize predictions
        self.visualize_predictions(X_test, y_test)
        
        return history
    
    def plot_training_history(self, history):
        """Plot training and validation metrics"""
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot MAE
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(self.checkpoint_path), 'training_history.png'))
        plt.close()
    
    def visualize_predictions(self, X_test, y_test, samples=5):
        """Visualize model predictions on test data"""
        # Get random samples
        indices = np.random.randint(0, len(X_test), samples)
        
        plt.figure(figsize=(15, samples * 3))
        
        for i, idx in enumerate(indices):
            img = X_test[idx]
            true_centers = self.denormalize_points(y_test[idx], img.shape)
            
            # Get model prediction
            pred = self.model.predict(np.expand_dims(img, axis=0))[0]
            pred_centers = self.denormalize_points(pred, img.shape)
            
            # Display results
            plt.subplot(samples, 2, 2*i+1)
            plt.imshow(img)
            for cx, cy in true_centers:
                plt.plot(cx, cy, 'go', markersize=8)
            plt.title('Ground Truth')
            plt.grid(True)
            
            plt.subplot(samples, 2, 2*i+2)
            plt.imshow(img)
            for cx, cy in pred_centers:
                plt.plot(cx, cy, 'ro', markersize=8)
            plt.title('Prediction')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(self.checkpoint_path), 'predictions.png'))
        plt.close()
    
    def denormalize_points(self, normalized_points, img_shape):
        """Convert normalized points back to pixel coordinates"""
        points = []
        for i in range(0, len(normalized_points), 2):
            x = int(normalized_points[i] * img_shape[1])
            y = int(normalized_points[i+1] * img_shape[0])
            points.append((x, y))
        return points
    
    def calculate_marker_accuracy(self, y_true, y_pred, threshold=0.05):
        """
        Calculate accuracy of marker detection based on normalized distance
        threshold: normalized distance threshold (0.05 = 5% of image dimension)
        """
        total_markers = len(y_true) // 2
        correct_markers = 0
        
        for i in range(total_markers):
            true_x = y_true[2 * i]
            true_y = y_true[2 * i + 1]
            pred_x = y_pred[2 * i]
            pred_y = y_pred[2 * i + 1]
            
            # Calculate Euclidean distance (normalized)
            distance = np.sqrt((true_x - pred_x) ** 2 + (true_y - pred_y) ** 2)
            
            if distance < threshold:
                correct_markers += 1
        
        return correct_markers / total_markers

    def evaluate_detailed(self, X_test, y_test):
        """Provide detailed evaluation of the model"""
        y_pred = self.model.predict(X_test)
        
        # Calculate overall metrics
        mse = np.mean((y_test - y_pred) ** 2)
        mae = np.mean(np.abs(y_test - y_pred))
        
        # Calculate accuracy at different thresholds
        accuracy_1 = np.mean([self.calculate_marker_accuracy(y_test[i], y_pred[i], 0.01) for i in range(len(y_test))])
        accuracy_5 = np.mean([self.calculate_marker_accuracy(y_test[i], y_pred[i], 0.05) for i in range(len(y_test))])
        accuracy_10 = np.mean([self.calculate_marker_accuracy(y_test[i], y_pred[i], 0.10) for i in range(len(y_test))])
        
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"Accuracy (1% threshold): {accuracy_1:.2%}")
        print(f"Accuracy (5% threshold): {accuracy_5:.2%}")
        print(f"Accuracy (10% threshold): {accuracy_10:.2%}")
        
        return {
            'mse': mse,
            'mae': mae,
            'accuracy_1': accuracy_1,
            'accuracy_5': accuracy_5,
            'accuracy_10': accuracy_10
        }

if __name__ == "__main__":
    # Set up paths
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "augmented")
    dataset_path = os.path.join(dataset_dir, "augmented_dataset.json")
    
    # Configure model options - removed model_type and input_size prompts
    batch_size = int(input("Enter batch size [default: 16]: ").strip() or "16")
    epochs = int(input("Enter epochs [default: 50]: ").strip() or "50")
    
    # Create and train the model
    detector = BlueMarkerDetector(
        batch_size=batch_size,
        epochs=epochs
    )
    
    detector.train(dataset_path)
    
    # Load the best model and perform detailed evaluation
    X_test, y_test = detector.load_dataset(dataset_path)
    _, X_test, _, y_test = train_test_split(X_test, y_test, test_size=0.2, random_state=42)
    
    print("\nLoading best model for evaluation...")
    detector.model = models.load_model(detector.checkpoint_path)
    metrics = detector.evaluate_detailed(X_test, y_test)
    
    print("\nModel saved to:", detector.checkpoint_path)
