import cv2
import numpy as np
import supervision as sv
from inference import get_model
import os
import time
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QSpinBox, QDoubleSpinBox, QComboBox
import sys

def process_video(video_path, output_path=None, model_name="yolov8n-pose-640", confidence=0.3, skip_frames=0):
    """Run pose detection on a video file and save the result."""
    # If output path not specified, create one based on input path
    if output_path is None:
        filename, ext = os.path.splitext(video_path)
        output_path = f"{filename}_annotated{ext}"
    
    # Load the pose detection model
    print(f"Loading {model_name} model...")
    model = get_model(model_name)
    
    # Initialize annotators
    vertex_annotator = sv.VertexAnnotator()
    box_annotator = sv.BoxAnnotator()
    
    # Initialize tracker for consistent IDs
    byte_tracker = sv.ByteTrack()
    byte_tracker.reset()
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize counters
    frame_count = 0
    processed_count = 0
    start_time = time.time()
    
    print(f"Processing video with {total_frames} frames at {fps} FPS...")
    
    # Define wrist parameters
    wrist_indices = [9, 10]  # Left and right wrist indices
    wrist_colors = [(0, 0, 255), (255, 0, 0)]  # Red for left, Blue for right
    circle_radius = 40  # Circle radius in pixels - increased
    circle_thickness = 1  # Increased thickness
    
    # Process video frames
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames if needed (for faster processing)
        if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
            continue
        
        processed_count += 1
        
        # Process frame with model
        result_raw = model.infer(frame, confidence=confidence)[0]
        
        # Convert results to supervision KeyPoints
        key_points = sv.KeyPoints.from_inference(result_raw)
        
        # Get detections for tracking and bounding boxes
        detections = sv.Detections.from_inference(result_raw)
        
        # Update tracking
        detections = byte_tracker.update_with_detections(detections)
        
        # Draw skeleton keypoints
        annotated_frame = frame.copy()
        
        # Use edge annotator for skeleton visualization
        edge_annotator = sv.EdgeAnnotator(
            color=sv.Color.GREEN,
            thickness=1
        )
        annotated_frame = edge_annotator.annotate(
            scene=annotated_frame,
            key_points=key_points
        )
        
        # Draw bounding boxes
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        
        # Add tracking IDs if available
        if detections.tracker_id is not None:
            for i, tracker_id in enumerate(detections.tracker_id):
                if i < len(detections.xyxy):
                    bbox = detections.xyxy[i]
                    x1, y1, x2, y2 = bbox
                    text = f"ID: {tracker_id}"
                    cv2.putText(
                        annotated_frame, text, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                    )
        
        # Draw circles around wrists
        if hasattr(key_points, 'xy') and key_points.xy is not None:
            # Loop through each person's data
            for person_idx in range(key_points.xy.shape[0]):
                # Process wrists for this person
                for idx, wrist_idx in enumerate(wrist_indices):
                    # Check if the keypoint is available
                    if wrist_idx < key_points.xy.shape[1]:
                        # Get coordinates for this person's wrist
                        wrist_x, wrist_y = key_points.xy[person_idx, wrist_idx]
                        
                        # Ensure coordinates are valid
                        if not (np.isnan(wrist_x) or np.isnan(wrist_y)):
                            wrist_x, wrist_y = int(wrist_x), int(wrist_y)
                            
                            # Check if coordinates are within image bounds
                            if (0 <= wrist_x < annotated_frame.shape[1] and 
                                0 <= wrist_y < annotated_frame.shape[0]):
                                
                                # Draw a circle around the wrist
                                cv2.circle(
                                    annotated_frame, 
                                    (wrist_x, wrist_y), 
                                    circle_radius, 
                                    wrist_colors[idx], 
                                    circle_thickness
                                )
        
        # Add progress information
        elapsed = time.time() - start_time
        fps_text = f"Processing: {processed_count}/{total_frames} frames ({processed_count/elapsed:.1f} FPS)"
        cv2.putText(
            annotated_frame, fps_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        
        # Write frame to output video
        writer.write(annotated_frame)
        
        # Optional: Display the frame during processing
        # Comment this out for faster processing
        cv2.imshow("Pose Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Print progress every 100 frames
        if processed_count % 100 == 0:
            percent_done = (frame_count / total_frames) * 100
            print(f"Processed {processed_count} frames ({percent_done:.1f}% complete)")
    
    # Release resources
    video.release()
    writer.release()
    cv2.destroyAllWindows()
    
    # Print summary
    total_time = time.time() - start_time
    print(f"Video processing complete!")
    print(f"Processed {processed_count} of {total_frames} frames in {total_time:.2f} seconds")
    print(f"Average processing speed: {processed_count/total_time:.2f} FPS")
    print(f"Output saved to: {output_path}")
    
    return output_path

class VideoProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 Pose Detection - Video Processing")
        self.setGeometry(100, 100, 500, 400)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Add model selection dropdown
        model_label = QLabel("Select Model:")
        layout.addWidget(model_label)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "yolov8n-pose-640", 
            "yolov8s-pose-640", 
            "yolov8m-pose-640", 
            "yolov8l-pose-640", 
            "yolov8x-pose-640"
        ])
        layout.addWidget(self.model_combo)
        
        # Add confidence threshold
        conf_label = QLabel("Confidence Threshold:")
        layout.addWidget(conf_label)
        
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.1, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.3)
        layout.addWidget(self.conf_spin)
        
        # Add skip frames option
        skip_label = QLabel("Skip Frames (0 = process all frames):")
        layout.addWidget(skip_label)
        
        self.skip_spin = QSpinBox()
        self.skip_spin.setRange(0, 10)
        self.skip_spin.setValue(0)
        layout.addWidget(self.skip_spin)
        
        # Status label
        self.status_label = QLabel("Select a video file to process")
        layout.addWidget(self.status_label)
        
        # Add file selection button
        self.select_button = QPushButton("Select Video File")
        self.select_button.clicked.connect(self.select_video)
        layout.addWidget(self.select_button)
        
        # Add process button
        self.process_button = QPushButton("Process Video")
        self.process_button.clicked.connect(self.process_video)
        self.process_button.setEnabled(False)
        layout.addWidget(self.process_button)
        
        # Video path
        self.video_path = None
    
    def select_video(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        
        if file_path:
            self.video_path = file_path
            self.status_label.setText(f"Selected: {os.path.basename(file_path)}")
            self.process_button.setEnabled(True)
    
    def process_video(self):
        if not self.video_path:
            return
        
        self.status_label.setText("Processing video... Please wait.")
        self.process_button.setEnabled(False)
        self.select_button.setEnabled(False)
        
        try:
            # Get parameters
            model_name = self.model_combo.currentText()
            confidence = self.conf_spin.value()
            skip_frames = self.skip_spin.value()
            
            # Process the video
            output_path = process_video(
                video_path=self.video_path,
                model_name=model_name,
                confidence=confidence,
                skip_frames=skip_frames
            )
            
            self.status_label.setText(f"Processing complete! Output: {os.path.basename(output_path)}")
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
        finally:
            self.process_button.setEnabled(True)
            self.select_button.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoProcessingApp()
    window.show()
    sys.exit(app.exec_()) 