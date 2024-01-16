import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Load the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained mask detection model (you'll need to have this trained or use a pre-existing one)
mask_model = load_model('my_model.keras')

# Function to detect mask on a face
def detect_mask(face):
    resized_face = cv2.resize(face, (224, 224))
    normalized_face = resized_face / 255.0
    reshaped_face = np.reshape(normalized_face, (1, 224, 224, 3))
    result = mask_model.predict(reshaped_face)
    return result[0][0] > 0.5  # Assuming the model outputs probabilities

# Read the image
image = cv2.imread('dsc08381small.jpg')

# Convert the image to grayscale (Haar cascades work on grayscale images)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

# Iterate through detected faces
for (x, y, w, h) in faces:
    face = image[y:y + h, x:x + w]  # Extract the face ROI
    has_mask = detect_mask(face)
    
    # Define the label and color for displaying mask/no mask
    label = "Mask" if has_mask else "No Mask"
    color = (0, 255, 0) if has_mask else (0, 0, 255)
    
    # Draw rectangles around the detected faces and label them
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

# Display the image with detected faces and mask detection results
cv2.imshow('Face Mask Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
