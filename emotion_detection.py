# emotion_detection.py

import cv2
import numpy as np
from keras.models import load_model
import os
from face_detection import detect_faces,get_face_landmarks

# Load the emotion detection model
emotion_classifier = load_model("models/emotionModel.hdf5")
emotion_offsets = (20, 40)

emotions = {
    0: {"emotion": "Angry", "save_path": "predicted/angry"},
    1: {"emotion": "Disgust", "save_path": "predicted/disgust"},
    2: {"emotion": "Fear", "save_path": "predicted/fear"},
    3: {"emotion": "Happy", "save_path": "predicted/happy"},
    4: {"emotion": "Sad", "save_path": "predicted/sad"},
    5: {"emotion": "Surprise", "save_path": "predicted/surprise"},
    6: {"emotion": "Neutral", "save_path": "predicted/neutral"}
}

def shape_points(shape):
    coords = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def rect_points(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def get_unique_filename(image_path, predicted_emotion, count=0):
    emotion_save_path = emotions[predicted_emotion]['save_path']
    filename = os.path.basename(image_path)
    filename_without_ext, ext = os.path.splitext(filename)

    # If the new filename already exists, increment the count and try again
    while os.path.exists(image_path):
        count += 1
        new_filename = f"{filename_without_ext} ({count}){ext}"
        image_path = os.path.join(emotion_save_path, new_filename)

    # If the count is 0, save the image with the original name
    if count == 0:
        save_file_path = os.path.join(emotion_save_path, filename)
    else:
        new_filename = f"{filename_without_ext} ({count}){ext}"
        save_file_path = os.path.join(emotion_save_path, new_filename)

    return save_file_path

def preprocess_face(face):
    # Preprocess the face image for emotion prediction
    face = cv2.resize(face, (emotion_classifier.input_shape[1], emotion_classifier.input_shape[2]))
    face = np.expand_dims(face, axis=0)
    face = face / 255.0
    return face

def predict_emotion(image_path):
    frame = cv2.imread(image_path)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    predicted_emotion = "Unknown"

    # Detect faces
    faces = detect_faces(gray_frame)
    for face_rect in faces:
        landmarks = get_face_landmarks(gray_frame, face_rect)
        (x, y, w, h) = rect_points(face_rect)

        # Extract the face ROI
        roi = gray_frame[y:y+h, x:x+w]

        try:
            # Preprocess the face for emotion prediction
            roi = preprocess_face(roi)

            # Perform emotion prediction
            emotion_prediction = emotion_classifier.predict(roi)
            emotion_label_arg = np.argmax(emotion_prediction)
            predicted_emotion = emotions[emotion_label_arg]['emotion']
            print("Predicted Emotion:", predicted_emotion)

            # Save the image with the original name or a unique name for subsequent images
            save_file_path = get_unique_filename(image_path, emotion_label_arg)
            cv2.imwrite(save_file_path, frame)
        except:
            continue

    return predicted_emotion
