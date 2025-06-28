from ultralytics import YOLO
import cv2

# Load YOLOv8 model (e.g., yolov8n.pt)
model = YOLO('yolov8n.pt')

# Open the default webcam
cap = cv2.VideoCapture(0)

# Target resize dimensions
target_width = 640
target_height = 480

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame before inference (optional)
    resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # Perform YOLOv8 inference
    results = model(resized_frame, verbose=False)

    # Draw bounding boxes on the frame
    annotated_frame = results[0].plot()

    # Display the result
    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and close everything
cap.release()
cv2.destroyAllWindows()
