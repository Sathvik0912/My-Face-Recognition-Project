# Face Recognition System (with Occlusion Handling and Supabase Integration)

This project is a real-time face recognition system built using Python, InsightFace, and OpenCV. It supports various modes such as occlusion handling, phone camera input, and Supabase-based login/logout tracking.

---

## 🔧 Features

- Real-time face recognition using webcam or phone stream.
- Handles occluded faces.
- Auto login/logout tracking with Supabase integration.
- Multiple versions for testing and deployment.

---

## 📁 Project Structure

```
sathvik0912-my-face-recognition-project/
├── finalreallyfinal2deploy.py     # Final deployment version (with Supabase)
├── idea.py                        # Initial idea prototype
├── includedoclusion.py            # Occlusion handling version
├── onlyrecog.py                   # Only recognition, no logging
├── onphone.py                     # Version for phone camera feed
├── reallyfinal1.py                # Prior final version
├── requirments.txt                # Python dependencies
└── dataset/                       # Folder containing face images
```

---

## 🚀 Setup Instructions

1. **Clone the repository** or download the code files.
2. **Create a Python environment** (optional but recommended):
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirments.txt
   ```
4. **Create a folder named `dataset`** in the project directory:
   ```bash
   mkdir dataset
   ```
   - Add images of known people inside this folder. The file name (without extension) will be treated as the person's name.
5. **Run the desired version of the script**:
   ```bash
   python finalreallyfinal2deploy.py
   ```
   - You can use `onlyrecog.py`, `includedoclusion.py`, etc., depending on your requirement.

---

## 🛠️ Requirements

- Python 3.11.9
- insightface
- opencv-python
- numpy
- onnxruntime
- opencv-contrib-python

---

## 📦 Dataset Notes

- All face images should be stored inside the `dataset/` folder.
- Images must be clear and ideally contain only one face.

---

## 🧠 Model

- Uses `insightface.app.FaceAnalysis` with the `buffalo_l` or `buffalo_s` model.
- Cosine similarity is used to match embeddings between live input and dataset images.

---

## ☁️ Supabase Integration

- The deployment version (`finalreallyfinal2deploy.py`) logs user login/logout events to Supabase.
- Make sure you update your Supabase URL and Key in the script.

---

## 📞 Phone Camera Usage

To use your phone as a camera, update the `url` in `onphone.py` to your IP webcam stream (e.g., via the IP Webcam Android app).

---

## 🖼️ Example Output

- Displays bounding boxes, face names, and confidence scores in real time.
- Press `q` to quit the webcam window.

---
