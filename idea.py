import os
import cv2
import insightface
import numpy as np

DATA_PATH = r"C:\Users\Hp\Documents\launchmodel\dataset"

face_recognition_model = insightface.app.FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'])
face_recognition_model.prepare(ctx_id=0, det_size=(640, 640))

known_face_encodings = []
known_face_names = []
not_found_faces = []

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
            known_face_encodings.append(embedding)
            known_face_names.append(os.path.splitext(file)[0])
    else:
        not_found_faces.append(file)

if not_found_faces:
    print("Files with no detected faces:")
    for nf in not_found_faces:
        print(nf)

print(f"Loaded {len(known_face_names)} faces from folder.")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 20)

threshold = 0.5

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    faces = face_recognition_model.get(frame)
    
    for face in faces:
        x, y, w, h = map(int, face.bbox)
        embedding = face.normed_embedding
        scores = [np.dot(embedding, ke) for ke in known_face_encodings]
        best_match_index = np.argmax(scores)
        best_match_score = scores[best_match_index]
        
        if best_match_score > threshold:
            best_match_name = known_face_names[best_match_index]
            accuracy = best_match_score * 100
        else:
            best_match_name = "Unknown"
            accuracy = 0
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{best_match_name} ({accuracy:.2f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if face.kps is not None:
            for (px, py) in face.kps:
                cv2.circle(frame, (int(px), int(py)), 3, (255, 0, 0), -1)
    
    cv2.imshow("Face Recognition Improved", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



