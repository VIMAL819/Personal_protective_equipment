from ultralytics import YOLO
import cv2

# Define the mandatory PPE items
mandatory_items = [
    'Dust Mask', 'Eye Wear', 'Glove', 'Protective Boots',
    'Protective Helmet', 'Safety Vest'
]

# Load your trained model
model = YOLO('runs/detect/train4/weights/best.pt')  # adjust the path if needed

# Load the image
img_path = "D:\PPE\download (1).jpeg"  # <-- replace with your image
image = cv2.imread(img_path)

# Run inference
results = model(img_path)[0]

# Get detected classes
detected_class_ids = [int(cls) for cls in results.boxes.cls]
detected_labels = [model.names[i] for i in detected_class_ids]

# Remove duplicates
detected_labels = list(set(detected_labels))

# Compare with mandatory list
missing_items = [item for item in mandatory_items if item not in detected_labels]

# Print results
print("✅ Detected PPE items:", detected_labels)
if missing_items:
    print("🛑 Missing PPE items:", missing_items)
else:
    print("✅ All mandatory PPE items are present.")

# Optional: Show the image with boxes
annotated_img = results.plot()
cv2.imshow("PPE Detection", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
