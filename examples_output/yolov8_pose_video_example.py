import cv2
import numpy as np
import supervision as sv
from inference import get_model
import os
import time

def process_video(video_path, output_path=None, model_name="yolov8n-pose-640", confidence=0.3, skip_frames=0):
    """Run pose detection on a video file and save the result."""
    # If output path not specified, create one based on input path
    if output_path is None:
        filename, ext = os.path.splitext(video_path)
        output_path = f"{filename}_pose_detection{ext}"
    
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
        annotated_frame = vertex_annotator.annotate(
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

if __name__ == "__main__":
    # Replace with your video path
    video_path = "path/to/your/video.mp4"
    
    # Model options: yolov8n-pose-640, yolov8s-pose-640, yolov8m-pose-640, yolov8l-pose-640, yolov8x-pose-640
    try:
        process_video(
            video_path=video_path,
            model_name="yolov8n-pose-640",
            confidence=0.3,
            skip_frames=0  # Set higher (e.g., 2) to process every 3rd frame for faster processing
        )
    except FileNotFoundError:
        print(f"Error: Video file '{video_path}' not found. Please provide a valid video path.")
    except Exception as e:
        print(f"Error: {e}") 