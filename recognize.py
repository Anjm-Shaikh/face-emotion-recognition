import cv2
import numpy as np
from keras.models import load_model
import pickle
import os

# Load models
face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
face_recognition_model = load_model('fac.h5')
emotion_model = load_model('emotion_model.h5')

# Load label names
with open('data/labels.p', 'rb') as f:
    labels = pickle.load(f)
    labels = {v: k for k, v in labels.items()}

# Load emoji images
emoji_folder = 'emoji'
emojis = {
    'happy': cv2.imread(os.path.join(emoji_folder, 'happy.png'), -1),
    'sad': cv2.imread(os.path.join(emoji_folder, 'sad.png'), -1),
    'angry': cv2.imread(os.path.join(emoji_folder, 'angry.png'), -1),
    'neutral': cv2.imread(os.path.join(emoji_folder, 'neutral.png'), -1)
}

# Emotion labels matching your model output
emotion_labels = ['angry', 'happy', 'neutral', 'sad']

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale, 1.3, 5)

    for (x, y, w, h) in faces:
        # FACE RECOGNITION
        face_img = grayscale[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (100, 100))
        face_input = face_img.reshape(1, 100, 100, 1) / 255.0
        pred = face_recognition_model.predict(face_input)
        person_name = labels[np.argmax(pred)]

        # EMOTION DETECTION
        emotion_pred = emotion_model.predict(face_input)
        emotion = emotion_labels[np.argmax(emotion_pred)]

        # DRAW EMOJI
        emoji_img = emojis[emotion]
        emoji_resized = cv2.resize(emoji_img, (w, h))
        for c in range(3):
            alpha = emoji_resized[:, :, 3] / 255.0
            frame[y:y+h, x:x+w, c] = (alpha * emoji_resized[:, :, c] +
                                      (1 - alpha) * frame[y:y+h, x:x+w, c])

        # DISPLAY TEXT
        cv2.putText(frame, f'{person_name} ({emotion})', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow('Face + Mood Detector 🤖', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
