import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import os
from collections import deque, Counter

# Load Haar cascade for face detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained emotion model
model = load_model('models/fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)

# Emotion labels from FER-2013 (ordered)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load emojis
emoji_images = {}
for label in emotion_labels:
    path = f'emojis/{label}.png'
    if os.path.exists(path):
        emoji = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        emoji_images[label] = emoji
    else:
        print(f"⚠️ Emoji for '{label}' not found in emojis/ folder")

# Overlay emoji beside face
def overlay_emoji(frame, emoji, x, y, w, h):
    size = 40  # match approx text size
    emoji = cv2.resize(emoji, (size, size))
    eh, ew = emoji.shape[:2]
    x_offset = x + w + 10
    y_offset = y

    if y_offset + eh > frame.shape[0]:
        y_offset = frame.shape[0] - eh
    if x_offset + ew > frame.shape[1]:
        x_offset = frame.shape[1] - ew

    for c in range(3):  # BGR channels
        alpha = emoji[:, :, 3] / 255.0
        frame[y_offset:y_offset+eh, x_offset:x_offset+ew, c] = (
            alpha * emoji[:, :, c] +
            (1 - alpha) * frame[y_offset:y_offset+eh, x_offset:x_offset+ew, c]
        )

# Set up webcam with HD resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("📸 Webcam started. Press 'q' to quit.")
recent_emotions = deque(maxlen=10)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)

        if np.sum(roi_gray) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = model.predict(roi, verbose=0)[0]
            confidence = np.max(prediction)
            label_index = np.argmax(prediction)
            label = emotion_labels[label_index]

            # Always append the predicted label and get the most common in last frames
            recent_emotions.append(label)
            most_common = Counter(recent_emotions).most_common(1)[0][0]

            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

            # Show emotion label
            cv2.putText(frame, most_common, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Show confidence percentage (optional — remove if you want)
            cv2.putText(frame, f"{confidence*100:.1f}%", (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Show emoji (always if available)
            if most_common in emoji_images:
                try:
                    overlay_emoji(frame, emoji_images[most_common], x, y, w, h)
                except:
                    pass
        else:
            cv2.putText(frame, 'No Face Found', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow('Real-Time Emotion Detection 😊', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Webcam closed.")
