from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')
img = cv2.imread('./images/20241009_200157.jpg')
resized_img = cv2.resize(img, (500, 500))
results = model(resized_img,show = True)
annotated_frame = results[0].plot()

cv2.imshow("Detection", annotated_frame)
cv2.waitKey(0)
