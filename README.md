# Real-Time Face Emotion Recognition 🎭

This is a Python-based real-time face emotion recognition system that detects human emotions using a webcam feed and overlays the predicted emotion and corresponding emoji.

## 📌 Project Summary

This application uses computer vision and deep learning techniques to recognize facial expressions and label them as one of the following emotions:

- Happy 😊
- Sad 😢
- Angry 😠
- Surprise 😲
- Fear 😨
- Disgust 🤢
- Neutral 😐

## 💻 Technologies Used

- Python 3
- OpenCV (for face detection)
- TensorFlow/Keras (for emotion classification)
- Pre-trained Mini-Xception model (FER-2013)
- NumPy, deque (for processing and smoothing)
- PNG emojis for visual overlay

## 🗂 Folder Structure

```
facedetection/
├── facedetection.py
├── models/
│   └── fer2013_mini_XCEPTION.102-0.66.hdf5
├── emojis/
│   └── happy.png, sad.png, angry.png, etc.
├── requirements.txt
└── README.md
```

## 🚀 How to Run

1. Clone this repository:
```bash
git clone https://github.com/yourusername/face-emotion-recognition.git
cd face-emotion-recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python facedetection.py
```



## 📄 License

MIT License © 2025 [Your Name]
