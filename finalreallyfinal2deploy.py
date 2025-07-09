import os
import cv2
import insightface
import numpy as np
from supabase import create_client
from datetime import datetime
##needed libs

SUPABASE_URL = "https://vymdywwttvflokzoxnxq.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZ5bWR5d3d0dHZmbG9rem94bnhxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDIxNzcxNTcsImV4cCI6MjA1Nzc1MzE1N30.oeOMZdXhP-omtkbQci2Ov2B_-i3pKDpPfxVNVzX6Qds"
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
##urls and keys

detection_tracker = {}
logout_timers = {}
relogin_freeze = {}
WAIT_TIME = 15
FREEZE_TIME = 15
##dictionarys qand timers

def insert_name_to_db(name_id, action):
    parts = name_id.split(',')
    name = parts[0].strip()
    user_id = parts[1].strip() if len(parts) > 1 else None
    timestamp = datetime.now().isoformat()
##takes name and ID as format seperated by ','
    
    if action == "login":
        existing_log = supabase_client.table("Login_Database").select("Logout").eq("Name", name).order("Login", desc=True).limit(1).execute()
        if existing_log.data and existing_log.data[0]["Logout"] is None:
            supabase_client.table("Login_Database").update({"Logout": timestamp}).eq("Name", name).is_("Logout", None).execute()
            print(f"{name} (ID: {user_id}) was (logged out) due to system restart.")
        else:
            supabase_client.table("Login_Database").insert({"Name": name, "ID Number": user_id, "Login": timestamp}).execute()
            print(f"{name} (ID: {user_id}) (logged in).")
    elif action == "logout":
        supabase_client.table("Login_Database").update({"Logout": timestamp}).eq("Name", name).is_("Logout", None).execute()
        print(f"{name} (ID: {user_id}) (logged out).")

DATA_PATH = r"C:\Users\Hp\Documents\launchmodel\dataset"
THRESHOLD_FULL = 0.35
THRESHOLD_OCCLUDED = 0.1
MIN_CONFIDENCE = 0.1
MIN_CONFIDENCE_OCCLUDED = 0.1

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
            known_face_encodings.append(embedding / np.linalg.norm(embedding))
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

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    faces = face_recognition_model.get(frame)
    current_time = datetime.now().timestamp()
    detected_names = set()

    for face in faces:
        x, y, w, h = map(int, face.bbox)
        embedding = face.normed_embedding / np.linalg.norm(face.normed_embedding)
        is_occluded = face.kps is None or len(face.kps) < 5
        threshold = THRESHOLD_OCCLUDED if is_occluded else THRESHOLD_FULL
        min_confidence = MIN_CONFIDENCE_OCCLUDED if is_occluded else MIN_CONFIDENCE
        scores = [np.dot(embedding, ke) for ke in known_face_encodings]
        best_match_index = np.argmax(scores)
        best_match_score = scores[best_match_index]
        best_match_name = known_face_names[best_match_index] if best_match_score > threshold else "Unknown"
        accuracy = (best_match_score + 1) / 2 * 100  

        if best_match_score < threshold or face.det_score < min_confidence:
            best_match_name = "Unknown"
            accuracy = 0 

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{best_match_name} ({accuracy:.2f}%)"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if face.kps is not None:
            for (px, py) in face.kps:
                cv2.circle(frame, (int(px), int(py)), 3, (255, 0, 0), -1)

        if best_match_name != "Unknown":
            if best_match_name in relogin_freeze and current_time < relogin_freeze[best_match_name]:
                continue
            if best_match_name not in detection_tracker:
                insert_name_to_db(best_match_name, "login")
                detection_tracker[best_match_name] = current_time + WAIT_TIME
                logout_timers[best_match_name] = current_time + WAIT_TIME

    for name in list(logout_timers.keys()):
        if current_time >= logout_timers[name]:
            faces_after_wait = face_recognition_model.get(frame)
            detected_names_after_wait = set()
            for face in faces_after_wait:
                scores = [np.dot(face.normed_embedding / np.linalg.norm(face.normed_embedding), ke) for ke in known_face_encodings]
                best_match_index = np.argmax(scores)
                best_match_score = scores[best_match_index]
                best_match_name = known_face_names[best_match_index] if best_match_score > THRESHOLD_FULL else "Unknown"
                detected_names_after_wait.add(best_match_name)
            if name not in detected_names_after_wait:
                continue  
            insert_name_to_db(name, "logout")
            relogin_freeze[name] = current_time + FREEZE_TIME
            del detection_tracker[name]
            del logout_timers[name]

    cv2.imshow("Face Recognition with Occlusion Handling", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()