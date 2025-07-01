from ultralytics import YOLO
import cv2
import os

model = YOLO("../models/yolov8n-face.pt")

def crop_faces_in_dataset(root_dir):
    person_folders = [os.path.join(root_dir, person) for person in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, person))]

    for person_folder in person_folders:
        print(f"📂 Processing: {person_folder}")
        images = os.listdir(person_folder)
        for img_name in images:
            img_path = os.path.join(person_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            results = model(img)[0]
            for box in results.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                face = img[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                resized = cv2.resize(face, (112, 112))
                cv2.imwrite(img_path, resized)  # Overwrite original 
crop_faces_in_dataset("./data_collected") 
