# Facial Emotion Recognition with Gradio and Deep Learning

This project is a **Facial Emotion Recognition App** that allows users to upload a video or use their webcam to detect the **dominant emotion** on faces throughout the video stream. It supports two emotion detection backends: `FER` and `DeepFace`, and is powered by OpenCV, Gradio, and deep learning models.

## 🔍 Features

- Detect emotions from faces in video files or live webcam feed
- Toggle between two detection backends: `FER` or `DeepFace`
- Real-time video processing with face detection and emotion overlay
- Option to skip alternate frames to improve speed
- Parallelized processing using all available CPU cores

## 🧠 Tech Stack

- **OpenCV** for image processing and drawing
- **DeepFace** and **FER** for emotion detection
- **Gradio** for building a friendly and interactive web UI
- **MTCNN** and Haar cascades for face detection
- **Multiprocessing** for performance improvement on video processing

---

## 🚀 How to Run Locally

### 1. Clone this Repository

```bash
git clone https://github.com/yourusername/faceai.git
cd faceai
```

### 2. Create and Activate a Virtual Environment (Optional but Recommended)

```bash
python3 -m venv env_faceAI
source env_faceAI/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
python main.py
```

Gradio will launch a local server and open your browser.

---

## 📂 Project Structure

```
faceAI/
├── app/
│   └── gradio_interface.py        # Gradio UI and interaction logic
├── detectors/
│   ├── emotion_detector.py        # FER and DeepFace emotion handling
│   └── face_detector.py           # Face detection via Haar/MTCNN
├── processing/
│   ├── utils.py                   # Helper to draw boxes and labels
│   ├── video_processor.py         # Main logic for uploaded video
│   └── webcam_processor.py        # Real-time webcam detection
├── flagged/                       # Empty for now; for flagged content
├── logs/                          # Log directory (can be used for debug)
├── processed_video.mp4            # Example output
├── main.py                        # Launch entry point
├── requirements.txt               # Dependencies
└── README.md                      # Project info
```

---

## 🧪 How it Works

- **Video Upload Tab:**

  - Choose a detector (FER or DeepFace)
  - Optionally skip alternate frames for speed
  - Process the video and view the result

- **Webcam Tab:**

  - Live webcam feed used for real-time emotion detection
  - Overlay dominant emotion on the face detected

---

## ✅ TODOs & Future Improvements

- Add GPU support (for faster inference with DeepFace)
- Save individual frame results with timestamps
- Improve error handling and logging
- Add unit tests

---

## 📜 License

This project is open-source under the MIT License.

---

## 🤝 Acknowledgements

- [DeepFace](https://github.com/serengil/deepface)
- [FER](https://github.com/justinshenk/fer)
- [Gradio](https://www.gradio.app/)
- [OpenCV](https://opencv.org/)

---

## 💡 Author

Made by Maria Duarte. Reach out for collaboration, improvement, or feedback!

