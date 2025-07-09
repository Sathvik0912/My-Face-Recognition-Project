##imports

import os
import cv2
import insightface
import numpy as np

#dataset path
DATA_PATH = r"C:\Users\Hp\Documents\launchmodel\dataset"

#model dec, rec
face_recognition_model = insightface.app.FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'])
face_recognition_model.prepare(ctx_id=0, det_size=(640, 640))

##storerooms 
known_face_encodings = []
known_face_names = []
not_found_faces = []

##reads the imgs in dataset
for file in os.listdir(DATA_PATH):
    img_path = os.path.join(DATA_PATH, file)
    image = cv2.imread(img_path)
    if image is None:                                 ##if not a file skips
        print(f"Skipping unreadable file: {file}")
        continue

##gets faces by using model
    faces = face_recognition_model.get(image)

##cam face in datasetfaces then applies norm-embedding
    if faces:
        for face in faces:
            embedding = face.normed_embedding  
            known_face_encodings.append(embedding / np.linalg.norm(embedding))  
            known_face_names.append(os.path.splitext(file)[0])
##cam face not in dataset
    else:
        not_found_faces.append(file)
        
##if faces not found 
if not_found_faces:
    print("Files with no detected faces:")
    for nf in not_found_faces:
        print(nf)

print(f"Loaded {len(known_face_names)} faces from folder.")
 
##cap = cv2.VideoCapture(0)            ## start cap
##cap.set(cv2.CAP_PROP_FPS, 20)                ## set 20 fps

threshold = 0.35              ##recog in 35%

url = "http://10.56.39.240:8080///video"  # Use the IP from your phone
cap = cv2.VideoCapture(url)
frame_count = 0

while cap.isOpened():     #in iteration
    ret, frame = cap.read()     ##reads frame
    if not ret:           ##if no frame detected 
        break            ##break
    frame_count += 1

    # Skips every 13 frames
    if frame_count % 13 != 0:
        continue
     #reset after 1000
    if frame_count >= 1000:
        frame_count = 0

    frame = cv2.flip(frame, 1)         #miror
    faces = face_recognition_model.get(frame)            #detect face using the model
    
    for face in faces:      ##iteration
        x, y, w, h = map(int, face.bbox)      ##gets face size
        embedding = face.normed_embedding        #gets facial feature like eye,nose,mouth ends
       
       ##comparing the similarities and b/w vertor from live and known faces
        scores = [np.dot(embedding, ke) for ke in known_face_encodings] 
        best_match_index = np.argmax(scores)
        best_match_score = scores[best_match_index]

        
        if best_match_score > threshold:     ## if score > 35
            best_match_name = known_face_names[best_match_index]     ##assign the kn face's name to best match
            accuracy = best_match_score * 100           ## accuracy calc
        
        ## else unknown and 0 accuracy
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



