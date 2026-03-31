"""
Vehicle Detection and Counting using YOLOv8
A simple computer vision project to detect and count vehicles in images.
"""

import cv2
from ultralytics import YOLO
import os


def detect_and_count_vehicles(input_image_path, output_image_path, confidence_threshold=0.5):
    """
    Detect and count vehicles in an image using YOLOv8.
    
    Args:
        input_image_path (str): Path to input image
        output_image_path (str): Path to save output image
        confidence_threshold (float): Minimum confidence for detection (0-1)
    
    Returns:
        dict: Dictionary containing vehicle counts by class
    """
    
    # Load YOLOv8 pre-trained model (nano version for faster processing)
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')  # Auto-downloads on first run
    
    # Read the input image
    print(f"Reading image: {input_image_path}")
    image = cv2.imread(input_image_path)
    
    if image is None:
        raise FileNotFoundError(f"Could not read image at: {input_image_path}")
    
    # COCO dataset vehicle class IDs and names
    vehicle_classes = {
        2: 'car',
        3: 'motorcycle', 
        5: 'bus',
        7: 'truck'
    }
    
    # Perform object detection
    print("Performing vehicle detection...")
    results = model(image, conf=confidence_threshold, verbose=False)
    
    # Initialize vehicle counter
    vehicle_count = {
        'car': 0,
        'motorcycle': 0,
        'bus': 0,
        'truck': 0
    }
    total_vehicles = 0
    
    # Process detection results
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            # Get class ID and confidence
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # Check if detected object is a vehicle
            if class_id in vehicle_classes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get vehicle type
                vehicle_type = vehicle_classes[class_id]
                
                # Increment counters
                vehicle_count[vehicle_type] += 1
                total_vehicles += 1
                
                # Define colors for different vehicle types (BGR format)
                colors = {
                    'car': (0, 255, 0),        # Green
                    'motorcycle': (255, 0, 0),  # Blue
                    'bus': (0, 165, 255),      # Orange
                    'truck': (0, 0, 255)       # Red
                }
                color = colors[vehicle_type]
                
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Prepare label with vehicle type and confidence
                label = f"{vehicle_type} {confidence:.2f}"
                
                # Calculate label size for background rectangle
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                
                # Draw label background
                cv2.rectangle(
                    image,
                    (x1, y1 - label_height - 10),
                    (x1 + label_width, y1),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )
    
    # Add total count and breakdown on the image
    overlay = image.copy()
    text_y_position = 30
    
    # Display total count
    total_text = f"Total Vehicles: {total_vehicles}"
    cv2.rectangle(overlay, (10, 10), (350, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    
    cv2.putText(
        image,
        total_text,
        (20, text_y_position),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )
    
    # Display class-wise count
    text_y_position += 30
    for vehicle_type, count in vehicle_count.items():
        if count > 0:
            count_text = f"{vehicle_type.capitalize()}: {count}"
            cv2.putText(
                image,
                count_text,
                (20, text_y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            text_y_position += 25
    
    # Save the output image
    cv2.imwrite(output_image_path, image)
    print(f"Output image saved to: {output_image_path}")
    
    # Return the counts
    vehicle_count['total'] = total_vehicles
    return vehicle_count


def main():
    """
    Main function to run vehicle detection.
    """
    
    print("=" * 50)
    print("Vehicle Detection and Counting System")
    print("=" * 50)
    
    # Define paths
    input_folder = "input"
    output_folder = "output"
    
    # Create folders if they don't exist
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    
    # Example: Process an image
    input_image = "images (4).jpg"
    input_path = os.path.join(input_folder, input_image)
    
    # Check if input image exists
    if not os.path.exists(input_path):
        print(f"\nError: Input image not found at '{input_path}'")
        print(f"Please place an image in the '{input_folder}' folder.")
        return
    
    # Generate output filename
    output_image = f"detected_{input_image}"
    output_path = os.path.join(output_folder, output_image)
    
    # Set confidence threshold (0.0 to 1.0)
    confidence_threshold = 0.5
    
    try:
        # Run vehicle detection
        counts = detect_and_count_vehicles(
            input_path,
            output_path,
            confidence_threshold
        )
        
        # Display results
        print("\n" + "=" * 50)
        print("DETECTION RESULTS")
        print("=" * 50)
        print(f"Total Vehicles Detected: {counts['total']}")
        print("\nBreakdown by Type:")
        print(f"  Cars:        {counts['car']}")
        print(f"  Motorcycles: {counts['motorcycle']}")
        print(f"  Buses:       {counts['bus']}")
        print(f"  Trucks:      {counts['truck']}")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")


if __name__ == "__main__":
    main()