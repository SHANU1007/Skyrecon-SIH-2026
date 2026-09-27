from ultralytics import YOLO
import cv2

# apna trained model load karo (pretrained yolov8n ki jagah)
model = YOLO('best.pt')

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    results = model(frame, conf=0.25)  # conf threshold adjust kar sakte ho
    annotated = results[0].plot()
    cv2.imshow('Trained Model - Human Detection', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()