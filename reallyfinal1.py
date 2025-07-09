import os
import cv2
import insightface
import numpy as np

DATA_PATH = r"C:\Users\Hp\Documents\launchmodel\dataset"
THRESHOLD_FULL = 0.35  # Increase threshold for normal faces  
THRESHOLD_OCCLUDED = 0.1 # Increase threshold for occluded faces  
MIN_CONFIDENCE = 0.1  # Increase confidence for full-face detection  
MIN_CONFIDENCE_OCCLUDED = 0.1  # Increase confidence for occluded faces  

# Load RetinaFace model
face_recognition_model = insightface.app.FaceAnalysis(name='buffalo_s', allowed_modules=['detection', 'recognition'])
face_recognition_model.prepare(ctx_id=0, det_size=(640, 640))

known_face_encodings = []
known_face_names = []
not_found_faces = []

# Load dataset
for file in os.listdir(DATA_PATH):
    img_path = os.path.join(DATA_PATH, file)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Skipping unreadable file: {file}")
        continue
    faces = face_recognition_model.get(image)
    if faces:
        for face in faces:
            embedding = face.normed_embedding
            known_face_encodings.append(embedding / np.linalg.norm(embedding))  # Normalize
            known_face_names.append(os.path.splitext(file)[0])
    else:
        not_found_faces.append(file)

# Print files with no detected faces
if not_found_faces:
    print("Files with no detected faces:")
    for nf in not_found_faces:
        print(nf)

print(f"Loaded {len(known_face_names)} faces from folder.")

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 20)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    faces = face_recognition_model.get(frame)

    for face in faces:
        x, y, w, h = map(int, face.bbox)
        embedding = face.normed_embedding / np.linalg.norm(face.normed_embedding)
        is_occluded = face.kps is None or len(face.kps) < 5
        threshold = THRESHOLD_OCCLUDED if is_occluded else THRESHOLD_FULL
        min_confidence = MIN_CONFIDENCE_OCCLUDED if is_occluded else MIN_CONFIDENCE
        scores = [np.dot(embedding, ke) for ke in known_face_encodings]
        best_match_index = np.argmax(scores)
        best_match_score = scores[best_match_index]
        best_match_name = known_face_names[best_match_index]
        accuracy = (best_match_score + 1) / 2 * 100  # Convert cosine similarity (-1 to 1) to percentage (0-100)

        # Detect unknown faces based on similarity and confidence
        if best_match_score < threshold or face.det_score < min_confidence:
            best_match_name = "Unknown"
            accuracy = 0

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display name and accuracy
        text = f"{best_match_name} ({accuracy:.2f}%)"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw key points if available
        if face.kps is not None:
            for (px, py) in face.kps:
                cv2.circle(frame, (int(px), int(py)), 3, (255, 0, 0), -1)

    cv2.imshow("Face Recognition Improved", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()