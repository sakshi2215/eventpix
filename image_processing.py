import cv2 as cv
import numpy as np
import mediapipe as mp

#Constants
FONTS = cv.FONT_HERSHEY_COMPLEX
# Face boundary indices
# Face boundary indices
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Left eye indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Right eye indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

map_face_mesh = mp.solutions.face_mesh

def get_eye_regions(image, mesh_coords):
    left_eye_points = [mesh_coords[p] for p in LEFT_EYE]
    right_eye_points = [mesh_coords[p] for p in RIGHT_EYE]

    # Extract bounding rectangles for left and right eyes
    left_eye_rect = cv.boundingRect(np.array(left_eye_points))
    right_eye_rect = cv.boundingRect(np.array(right_eye_points))

    return left_eye_rect, right_eye_rect

def preprocess_eye_image(eye_image):
    # Resize the eye image to (224, 224) and normalize
    eye_image = cv.resize(eye_image, (224, 224))
    eye_image = np.expand_dims(eye_image, axis=0)
    eye_image = eye_image / 255.0

    return eye_image

def predict_eye_state(model, eye_image):
    # Make predictions for the eye image
    prediction = model.predict(eye_image)

    return prediction[0]

def process_image(image_path, model):
    image = cv.imread(image_path)
    if image is None:
        return None

    with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        # Convert image from RGB to BGR
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        results = face_mesh.process(rgb_image)
        if results.multi_face_landmarks:
            mesh_coords = [(int(point.x * image.shape[1]), int(point.y * image.shape[0])) for point in results.multi_face_landmarks[0].landmark]

            # Extract the bounding rectangle for the left and right eye
            left_eye_rect = cv.boundingRect(np.array([mesh_coords[p] for p in LEFT_EYE]))
            right_eye_rect = cv.boundingRect(np.array([mesh_coords[p] for p in RIGHT_EYE]))

             # Scaling factors to increase the height and width of bounding boxes
            height_scale = 1.5
            width_scale = 1.3

            # Calculate the new height and width of the bounding boxes
            new_left_eye_height = int(left_eye_rect[3] * height_scale)
            new_left_eye_width = int(left_eye_rect[2] * width_scale)
            new_right_eye_height = int(right_eye_rect[3] * height_scale)
            new_right_eye_width = int(right_eye_rect[2] * width_scale)

            # Calculate the new (x, y) positions for the bounding boxes
            new_left_eye_x = max(0, left_eye_rect[0] - int((new_left_eye_width - left_eye_rect[2]) / 2))
            new_left_eye_y = max(0, left_eye_rect[1] - int((new_left_eye_height - left_eye_rect[3]) / 2))
            new_right_eye_x = max(0, right_eye_rect[0] - int((new_right_eye_width - right_eye_rect[2]) / 2))
            new_right_eye_y = max(0, right_eye_rect[1] - int((new_right_eye_height - right_eye_rect[3]) / 2))

            # Adjust the height and width of the bounding boxes
            new_left_eye_height = min(image.shape[0] - new_left_eye_y, new_left_eye_height)
            new_left_eye_width = min(image.shape[1] - new_left_eye_x, new_left_eye_width)
            new_right_eye_height = min(image.shape[0] - new_right_eye_y, new_right_eye_height)
            new_right_eye_width = min(image.shape[1] - new_right_eye_x, new_right_eye_width)

            # Crop the left and right eye regions from the original image using the new bounding box sizes
            left_eye_image = image[new_left_eye_y:new_left_eye_y + new_left_eye_height,
                                   new_left_eye_x:new_left_eye_x + new_left_eye_width]
            right_eye_image = image[new_right_eye_y:new_right_eye_y + new_right_eye_height,
                                    new_right_eye_x:new_right_eye_x + new_right_eye_width]

            # Resize the left and right eye images to (224, 224) and preprocess
            final_left_eye = cv.resize(left_eye_image, (224, 224))
            final_left_eye = np.expand_dims(final_left_eye, axis=0)
            final_left_eye = final_left_eye / 255.0

            final_right_eye = cv.resize(right_eye_image, (224, 224))
            final_right_eye = np.expand_dims(final_right_eye, axis=0)
            final_right_eye = final_right_eye / 255.0

            # Make the predictions for the left and right eyes
            left_eye_prediction = model.predict(final_left_eye)
            right_eye_prediction = model.predict(final_right_eye)

            # Determine the result based on the predictions
            if left_eye_prediction[0] > 0.5 and right_eye_prediction[0] > 0.5:
                result = " Open Eyes"
            elif left_eye_prediction[0] > 0.5 or right_eye_prediction[0] > 0.5:
                result = " Open Eyes"
            else:
                result = "Closed Eyes"

            print("Result:", result)

            return result

        return None
