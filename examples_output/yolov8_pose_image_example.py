import cv2
import numpy as np
import supervision as sv
from inference import get_model
import matplotlib.pyplot as plt

# Set to "skeleton" or "labels" to choose visualization style
VISUALIZATION_MODE = "labels"

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
    
    # Debug: Print keypoints structure
    print(f"KeyPoints type: {type(key_points)}")
    print(f"KeyPoints length: {len(key_points)}")
    if len(key_points) > 0:
        print(f"First keypoint set type: {type(key_points[0])}")
        print(f"First keypoint set attributes: {dir(key_points[0])}")
        if hasattr(key_points, 'xy'):
            print(f"KeyPoints.xy shape: {key_points.xy.shape if hasattr(key_points.xy, 'shape') else 'N/A'}")
            print(f"First few keypoint coordinates: {key_points.xy[0, :5] if hasattr(key_points.xy, '__getitem__') else key_points.xy}")
    
    # Define keypoint labels and colors
    LABELS = [
        "nose", "left eye", "right eye", "left ear",
        "right ear", "left shoulder", "right shoulder", "left elbow",
        "right elbow", "left wrist", "right wrist", "left hip",
        "right hip", "left knee", "right knee", "left ankle",
        "right ankle"
    ]

    COLORS = [
        "#FF6347", "#FF6347", "#FF6347", "#FF6347",
        "#FF6347", "#FF1493", "#00FF00", "#FF1493",
        "#00FF00", "#FF1493", "#00FF00", "#FFD700",
        "#00BFFF", "#FFD700", "#00BFFF", "#FFD700",
        "#00BFFF"
    ]
    COLORS = [sv.Color.from_hex(color_hex=c) for c in COLORS]
    
    # Create a copy of the image for annotation
    annotated_image = image.copy()
    
    # Debug: Print image shape
    print(f"Image shape: {annotated_image.shape}")
    
    # Draw skeleton keypoints
    if len(key_points) > 0:
        if VISUALIZATION_MODE == "labels":
            # First draw the skeleton connections
            vertex_annotator = sv.VertexAnnotator()
            annotated_image = vertex_annotator.annotate(
                scene=annotated_image,
                key_points=key_points
            )
            
            # Then add the labeled keypoints
            vertex_label_annotator = sv.VertexLabelAnnotator(
                color=COLORS,
                text_color=sv.Color.BLACK,
                border_radius=5
            )
            annotated_image = vertex_label_annotator.annotate(
                scene=annotated_image,
                key_points=key_points,
                labels=LABELS
            )
        elif VISUALIZATION_MODE == "skeleton":
            # Use edge annotator for pure skeleton visualization
            edge_annotator = sv.EdgeAnnotator(
                color=sv.Color.GREEN,
                thickness=5
            )
            annotated_image = edge_annotator.annotate(
                scene=annotated_image,
                key_points=key_points
            )
        
        # Draw circles around left and right wrists
        # Wrist indices: 9 for left wrist, 10 for right wrist
        wrist_indices = [9, 10]  # Left and right wrist indices
        wrist_colors = [(0, 0, 255), (255, 0, 0)]  # Red for left, Blue for right
        circle_radius = 30  # Circle radius in pixels
        circle_thickness = 3
        
        print("\n--- WRIST DETECTION DEBUG ---")
        print(f"Number of detected persons: {len(key_points)}")
        
        # Access the keypoint data from the overall object, not individual items
        if hasattr(key_points, 'xy') and key_points.xy is not None:
            print(f"Keypoints xy shape: {key_points.xy.shape}")
            
            # Loop through each person's data
            for person_idx in range(key_points.xy.shape[0]):
                print(f"\nPerson {person_idx+1} keypoints:")
                
                # Process wrists for this person
                for idx, wrist_idx in enumerate(wrist_indices):
                    wrist_name = "Left wrist" if idx == 0 else "Right wrist"
                    
                    # Check if the keypoint is available
                    if wrist_idx < key_points.xy.shape[1]:
                        # Get coordinates for this person's wrist
                        wrist_x, wrist_y = key_points.xy[person_idx, wrist_idx]
                        print(f"  {wrist_name} (idx {wrist_idx}) raw coords: ({wrist_x}, {wrist_y})")
                        
                        # Ensure coordinates are valid
                        if not (np.isnan(wrist_x) or np.isnan(wrist_y)):
                            wrist_x, wrist_y = int(wrist_x), int(wrist_y)
                            print(f"  {wrist_name} valid coords: ({wrist_x}, {wrist_y})")
                            
                            # Check if coordinates are within image bounds
                            if (0 <= wrist_x < annotated_image.shape[1] and 
                                0 <= wrist_y < annotated_image.shape[0]):
                                print(f"  {wrist_name} in bounds, drawing circle with radius {circle_radius}, color {wrist_colors[idx]}")
                                
                                # Draw a circle around the wrist
                                cv2.circle(
                                    annotated_image, 
                                    (wrist_x, wrist_y), 
                                    circle_radius, 
                                    wrist_colors[idx], 
                                    circle_thickness
                                )
                            else:
                                print(f"  {wrist_name} OUT OF BOUNDS: ({wrist_x}, {wrist_y}) for image shape {annotated_image.shape}")
                        else:
                            print(f"  {wrist_name} has invalid coordinates (NaN)")
                    else:
                        print(f"  {wrist_name} index {wrist_idx} out of range")
        else:
            print("No 'xy' attribute found on key_points object or xy is None")
        
        print("--- END WRIST DETECTION DEBUG ---\n")
        
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
    image_path = "/Users/chrissmith/Desktop/pp.png"
    
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