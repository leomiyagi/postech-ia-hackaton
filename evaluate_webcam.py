from ultralytics import YOLO
import cv2
import time
import argparse

def process_webcam(model_path, conf_threshold=0.25, show_fps=True):
    """
    Process webcam feed with YOLOv8 model in real-time
    
    Args:
        model_path: Path to the trained model
        conf_threshold: Confidence threshold for detections
        show_fps: Whether to show FPS counter
    """
    # Load the model
    print("Loading model...")
    model = YOLO(model_path)
    
    # Initialize webcam
    print("Starting webcam...")
    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detection', 640, 640)
    cv2.startWindowThread()

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise ValueError("Could not open webcam")
    
    # Get webcam properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("Press 'q' to quit")
    
    # Initialize FPS counter
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model(frame, conf=conf_threshold)[0]
        
        # Draw results on frame
        annotated_frame = results.plot()
        
        # Calculate and display FPS
        frame_count += 1
        if frame_count % 30 == 0:  # Update FPS every 30 frames
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
        
        if show_fps:
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show number of detections
        num_detections = len(results.boxes)
        if num_detections > 0:
            text_color = (0, 0, 255)  # Red for detections
        else:
            text_color = (0, 255, 0)  # Green for no detections
            
        cv2.putText(annotated_frame, f"Detections: {num_detections}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        
        # Display confidence for each detection
        for i, box in enumerate(results.boxes):
            conf = box.conf.item()
            y_pos = 120 + (i * 40)
            cv2.putText(annotated_frame, f"Confidence {i+1}: {conf:.2%}", 
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        
        # Display the frame
        cv2.imshow('Object Detection', annotated_frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    print("\nSession Statistics:")
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {frame_count/(time.time() - start_time):.1f}")

def main():
    parser = argparse.ArgumentParser(description='Process webcam feed with YOLOv8 model')
    parser.add_argument('--model', default='best_finetunned.pt',
                      help='Path to the trained model')
    parser.add_argument('--conf', type=float, default=0.25,
                      help='Confidence threshold (default: 0.25)')
    parser.add_argument('--no-fps', action='store_false', dest='show_fps',
                      help='Hide FPS counter')
    
    args = parser.parse_args()
    
    try:
        process_webcam(args.model, args.conf, args.show_fps)
    except Exception as e:
        print(f"Error processing webcam feed: {str(e)}")
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 