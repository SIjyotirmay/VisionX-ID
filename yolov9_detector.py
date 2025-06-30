# from ultralytics import YOLO
# import cv2

#    # or yolov9n.pt

# img = cv2.imread('./images/20241009_200157.jpg')
# resized_img = cv2.resize(img, (500, 500))

# # Detect face
# face_results = face_model(resized_img)

# # Detect body
# body_results = body_model(resized_img)

# cv2.imshow("face Detection", face_results)
# cv2.imshow("Person Detection", body_results)
# cv2.waitKey(1)



from ultralytics import YOLO
import cv2
import cvzone
import math


cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("./Videos")

cap.set(3,1000)
cap.set(4,720)
# model = YOLO('yolov8n.pt').to('cuda')
face_model = YOLO('yolov8n-face.pt')
body_model = YOLO('yolov8n.pt')

# classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#     "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#     "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#     "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#     "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#     "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#     "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#     "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#     "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#     "teddy bear", "hair drier", "toothbrush"
# ]



while True:
    sucess, img = cap.read()
    results = face_model(img, stream=True)
    for r in results:
        boxes  = r.boxes
        for box in boxes:

            #bounding box
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            #print(x1,y1,x2,y2)
            #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),thickness=3)
            w ,h = x2-x1, y2-y1
            #bbox = int(x1),int(y1),int(w),int(h)
            cvzone.cornerRect(img,(x1,y1,w,h))

            #confidence
            conf = math.ceil((box.conf[0]*100)) /100
            #print(conf)
            #cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(30,y1)))
            
            #class name
            cls  = int(box.cls[0])
            cvzone.putTextRect(img, f'{conf}',(max(0,x1),max(30,y1)) )



    cv2.imshow("image",img)
    cv2.waitKey(1)