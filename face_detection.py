# face_detection.py

import dlib

# Initialize face detector and predictor
face_detector = dlib.get_frontal_face_detector()
face_landmark_predictor = dlib.shape_predictor("models/dlib/shape_predictor_68_face_landmarks.dat")

def detect_faces(image):
    # Perform face detection and return the rectangles for detected faces
    return face_detector(image, 0)

def get_face_landmarks(image, face_rect):
    # Given an image and a face rectangle, return the facial landmarks
    shape = face_landmark_predictor(image, face_rect)
    return [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)]
