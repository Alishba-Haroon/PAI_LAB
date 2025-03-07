import cv2
import numpy as np
import os

image_path = "download.jpeg" 
config_path = "yolov3.cfg" 
weights_path = "yolov3.weights" 
classes_path = "yolov3.txt" 
with open(classes_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
image = cv2.imread(image_path)
if image is None:
    print("Error: Image not found!")
    exit()
Height, Width = image.shape[:2]
scale = 0.00392
net = cv2.dnn.readNet(weights_path, config_path)
blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
outs = net.forward(output_layers)
class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4
for out in outs:
    for detection in out:
        scores = detection[5:] 
        class_id = np.argmax(scores)
        confidence = scores[class_id] 
        if confidence > conf_threshold:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
if len(indices) > 0:
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        class_id = class_ids[i]
        confidence = confidences[i]
        label = f"{classes[class_id]} ({int(confidence * 100)}%)"
        color = [int(c) for c in COLORS[class_id]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text_x, text_y = x, max(y - 10, 10) 
        cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
output_path = "detected_objects.jpg"
cv2.imwrite(output_path, image)
print(f"✅ Detection complete! Image saved as '{output_path}'")
cv2.imshow("YOLO Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()