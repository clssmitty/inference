import cv2
import numpy as np
import supervision as sv
from inference import get_model
import matplotlib.pyplot as plt

def process_image(image_path, model_name="yolov8n-pose-640", confidence=0.3):
    """Run pose detection on a single image and visualize results."""
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
    
    # Load the pose detection model
    model = get_model(model_name)
    
    # Run inference
    result_raw = model.infer(image, confidence=confidence)[0]
    
    # Convert results to supervision KeyPoints
    key_points = sv.KeyPoints.from_inference(result_raw)
    
    # Initialize annotators
    vertex_annotator = sv.VertexAnnotator()
    
    # Create a copy of the image for annotation
    annotated_image = image.copy()
    
    # Draw skeleton keypoints
    if len(key_points) > 0:
        annotated_image = vertex_annotator.annotate(
            scene=annotated_image,
            key_points=key_points
        )
        
        # Get detections for bounding boxes
        detections = sv.Detections.from_inference(result_raw)
        
        # Draw bounding boxes
        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(
            scene=annotated_image,
            detections=detections
        )
        
        # Print information about the detected poses
        print(f"Found {len(key_points)} persons with pose keypoints")
        for i, keypoint_set in enumerate(key_points):
            if hasattr(keypoint_set, 'confidence') and keypoint_set.confidence is not None:
                print(f"Person {i+1} - Confidence: {keypoint_set.confidence:.2f}")
    else:
        print("No poses detected in the image")
    
    # Display the results
    plt.figure(figsize=(12, 12))
    plt.imshow(annotated_image)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the annotated image
    output_path = image_path.replace('.jpg', '_annotated.jpg').replace('.png', '_annotated.png')
    plt.savefig(output_path)
    print(f"Annotated image saved to: {output_path}")
    
    plt.show()
    
    return annotated_image, key_points

if __name__ == "__main__":
    # Provide an image path to run pose detection on
    image_path = "/Users/chrissmith/Downloads/SporfieVideos/sportpickle_court.png"
    
    # You can choose from different model sizes:
    # - yolov8n-pose-640 (nano - fastest, less accurate)
    # - yolov8s-pose-640 (small)
    # - yolov8m-pose-640 (medium)
    # - yolov8l-pose-640 (large)
    # - yolov8x-pose-640 (extra-large - slowest, most accurate)
    
    try:
        process_image(
            image_path=image_path,
            model_name="yolov8n-pose-640",
            confidence=0.1
        )
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found. Please provide a valid image path.")
    except Exception as e:
        print(f"Error: {e}") 