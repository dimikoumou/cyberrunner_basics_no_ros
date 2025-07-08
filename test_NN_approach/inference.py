import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import argparse
import time

class MarkerPredictor:
    def __init__(self, model_path, input_shape=(224, 224, 3)):
        """Initialize the marker predictor with a trained model"""
        self.input_shape = input_shape
        self.model = models.load_model(model_path)
        print(f"Model loaded from {model_path}")
    
    def preprocess_image(self, frame):
        """Preprocess image for model input"""
        # Save original dimensions
        self.original_height, self.original_width = frame.shape[:2]
        
        # Resize
        img_resized = cv2.resize(frame, (self.input_shape[0], self.input_shape[1]))
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img_normalized = img_rgb / 255.0
        
        return img_normalized
    
    def denormalize_points(self, normalized_points):
        """Convert normalized points back to original image coordinates"""
        points = []
        for i in range(0, len(normalized_points), 2):
            x = int(normalized_points[i] * self.original_width)
            y = int(normalized_points[i+1] * self.original_height)
            points.append((x, y))
        return points
    
    def predict_markers(self, frame):
        """Predict marker positions from a single frame"""
        # Preprocess the image
        processed_img = self.preprocess_image(frame)
        
        # Make prediction
        prediction = self.model.predict(np.expand_dims(processed_img, axis=0), verbose=0)[0]
        
        # Convert normalized coordinates back to pixel coordinates
        marker_positions = self.denormalize_points(prediction)
        
        return marker_positions
    
    def process_video(self, video_path, output_path=None, show_display=True):
        """Process a video and detect markers in each frame"""
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create video writer if needed
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_idx = 0
        processing_times = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Start time measurement
            start_time = time.time()
            
            # Predict markers
            markers = self.predict_markers(frame)
            
            # End time measurement
            end_time = time.time()
            processing_time = end_time - start_time
            processing_times.append(processing_time)
            
            # Draw markers on the frame
            for i, (x, y) in enumerate(markers):
                cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)  # Red circle
                cv2.circle(frame, (x, y), 6, (255, 255, 255), -1)  # White circle
                cv2.putText(frame, str(i), (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Add processing time info
            cv2.putText(frame, f"Frame: {frame_idx} | Time: {processing_time:.3f}s", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Write frame to output video if needed
            if output_path:
                out.write(frame)
            
            # Show frame if needed
            if show_display:
                cv2.imshow('Marker Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_idx += 1
        
        # Release resources
        cap.release()
        if output_path:
            out.release()
        if show_display:
            cv2.destroyAllWindows()
        
        # Print statistics
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            print(f"Processed {frame_idx} frames")
            print(f"Average processing time: {avg_time:.3f}s per frame ({1/avg_time:.2f} FPS)")
    
    def process_image(self, image_path, output_path=None, show_display=True):
        """Process a single image and detect markers"""
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image {image_path}")
            return
        
        # Predict markers
        markers = self.predict_markers(frame)
        
        # Draw markers on the frame
        for i, (x, y) in enumerate(markers):
            cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)  # Red circle
            cv2.circle(frame, (x, y), 6, (255, 255, 255), -1)  # White circle  
            cv2.putText(frame, str(i), (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Write processed image if needed
        if output_path:
            cv2.imwrite(output_path, frame)
            print(f"Saved processed image to {output_path}")
        
        # Show image if needed
        if show_display:
            cv2.imshow('Marker Detection', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return markers
    
    def run_webcam(self, camera_id=0, output_path=None):
        """Run live detection on webcam feed"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 20  # Set a reasonable FPS for recording
        
        # Create video writer if needed
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        print("Press 'q' to quit...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Predict markers
            markers = self.predict_markers(frame)
            
            # Draw markers on the frame
            for i, (x, y) in enumerate(markers):
                cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)
                cv2.circle(frame, (x, y), 6, (255, 255, 255), -1)
                cv2.putText(frame, str(i), (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Write frame to output video if needed
            if output_path:
                out.write(frame)
            
            # Show frame
            cv2.imshow('Webcam Marker Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Blue Marker Detection Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.h5 file)')
    parser.add_argument('--input', type=str, help='Path to input image or video')
    parser.add_argument('--output', type=str, help='Path to save output image or video')
    parser.add_argument('--size', type=int, default=224, help='Input size for the model')
    parser.add_argument('--webcam', type=int, default=None, help='Webcam ID for live detection')
    parser.add_argument('--no-display', action='store_true', help='Disable display window')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = MarkerPredictor(args.model, input_shape=(args.size, args.size, 3))
    
    # Check if webcam mode is selected
    if args.webcam is not None:
        predictor.run_webcam(args.webcam, args.output)
        return
    
    # Check if input is provided
    if args.input is None:
        print("Error: Input file or webcam ID must be provided.")
        return
    
    # Determine if input is image or video
    if os.path.isfile(args.input):
        _, ext = os.path.splitext(args.input)
        if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            predictor.process_image(args.input, args.output, not args.no_display)
        elif ext.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
            predictor.process_video(args.input, args.output, not args.no_display)
        else:
            print("Error: Unsupported file format.")
    else:
        print("Error: Input file does not exist.")


if __name__ == "__main__":
    main()
