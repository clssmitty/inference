import cv2
import numpy as np
import supervision as sv
import inference
from inference import Stream, get_model

# Initialize annotators for pose visualization
vertex_annotator = sv.VertexAnnotator()
box_annotator = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(['#FF8C00']),
    thickness=2
)
# Initialize tracker for consistent pose IDs across frames
byte_tracker = sv.ByteTrack()

def process_frame(detections, image):
    """Process each frame with pose detection and tracking."""
    # Convert the raw detections to supervision KeyPoints format
    key_points = sv.KeyPoints.from_inference(detections)
    
    # Convert to Detections format for tracking
    detections = sv.Detections.from_inference(detections)
    
    # Update tracks with new detections
    detections = byte_tracker.update_with_detections(detections)
    
    # Create a copy of the frame for annotation
    annotated_frame = image.copy()
    
    # Draw the pose keypoints on the frame
    annotated_frame = vertex_annotator.annotate(
        scene=annotated_frame,
        key_points=key_points
    )
    
    # Draw bounding boxes and tracking IDs
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame,
        detections=detections
    )
    
    # Add tracking IDs as text if available
    if detections.tracker_id is not None:
        for i, tracker_id in enumerate(detections.tracker_id):
            bbox = detections.xyxy[i]
            x1, y1, x2, y2 = bbox
            text = f"ID: {tracker_id}"
            cv2.putText(
                annotated_frame, text, (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
    
    # Display the annotated frame
    cv2.imshow("YOLOv8 Pose Tracking", annotated_frame)
    cv2.waitKey(1)

def main():
    """Run pose detection on webcam stream."""
    # Choose YOLOv8 pose model size (options: n, s, m, l, x)
    model_name = "yolov8n-pose-640"
    
    print(f"Starting pose detection with {model_name}...")
    print("Press 'q' to quit")
    
    try:
        # For webcam streaming with real-time processing
        Stream(
            source="webcam",
            model=model_name,
            output_channel_order="BGR",
            use_main_thread=True,
            on_prediction=process_frame,
            use_bytetrack=True
        )
    except KeyboardInterrupt:
        print("Stopping pose detection...")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 