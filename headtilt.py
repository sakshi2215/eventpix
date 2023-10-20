# -*- coding: utf-8 -*-
# @Author: Dastan_Alam
# @Date:   09-07-2023 01:36:08 AM       01:36:08
# @Last Modified by:   Dastan_Alam
# @Last Modified time: 20-07-2023 11:52:44 PM       23:52:44


from flask import Flask, render_template, request, Response
import cv2
import mediapipe as mp
import numpy as np
import time
import base64

app = Flask(__name__, template_folder="template")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return 'No file selected!'

    file = request.files['image']
    image = cv2.imdecode(np.fromstring(
        file.read(), np.uint8), cv2.IMREAD_COLOR)

    if image is not None:
        # Process the image here]
        result, direction = detect_head_tilt(image)
        # Display the direction on the image
        # cv2.putText(image, direction, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the image

        # cv2.imshow('Head Tilt Detection', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Convert the OpenCV image to base64 format for rendering in HTML
        _, encoded_image = cv2.imencode('.jpg', result)
        image_base64 = base64.b64encode(encoded_image).decode('utf-8')

        return render_template('local_r.html', direction=direction, image_base64=image_base64)
    else:
        return 'Failed to load the image.'


def detect_head_tilt(frame):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # # Replace 'image_path' with the path to your local image file
    # image_path = 'C:\\data\\lrft and witre\\head\\data.shape_predictor_68_face_landmarks\\master\\dataset\\1 (5).jpg'

    # # Read the image from the local file
    # image = cv2.imread(image_path)

    # if image is None:
    #     print("Error: Unable to read the image from the specified path.")
    #     exit()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space BGR to RGB

    # image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    start = time.time()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False

    # Get the results
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True
    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D coordinates
                    face_2d.append([x, y])

                    # Get the 3D coordinates
                    face_3d.append([x, y, lm.z])

            # Convert it to a numpy array
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # Camera matrix
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            # The Distortion Parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(
                face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation angles
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # See user head tilting
            if y < -10:
                text = "Looking left"
            elif y > 10:
                text = "Looking Right"
            elif x < -10:
                text = "Looking Down"
            elif x > 10:
                text = "Looking Up"
            else:
                text = "Forward"

            # Display nose direction

            nose_3d_projection, jacobian = cv2.projectPoints(
                nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
            cv2.line(image, p1, p2, (255, 0, 0), 3)

            cv2.putText(image, text, (0, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, "x: " + str(np.round(x, 2)), (0, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "y: " + str(np.round(y, 2)), (0, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "z: " + str(np.round(z, 2)), (0, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        end = time.time()
        tt = end-start
        print("total time",tt,"sec")
    else:
        image = frame
        text = "not face"
        cv2.putText(image, text, (0, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    # Display the image with annotations
    # cv2.imshow('Head Pose Estimation', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return (image, text)


@app.route('/start_program')
def live():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        start = time.time()

        # flip the image horizontally for a later selfie-view display
        # also convert the color space BGR to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To imporve performance
        image.flags.writeable = False

        # Get  the results
        results = face_mesh.process(image)

        # to imporve performance
        image.flags.writeable = True
        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z*3000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D coordinates
                        face_2d.append([x, y])

                        # Get the 3D coordinates
                        face_3d.append([x, y, lm.z])

                # convert it to numpy array
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                # Camera materix
                focal_length = 1*img_w
                cam_matrix = np.array([[focal_length, 0, img_h/2],
                                       [0, focal_length, img_w/2],
                                       [0, 0, 1]])

                # The Distortation Parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve pnp
                success, rot_vec, trans_vec = cv2.solvePnP(
                    face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # get the y rotation angles
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                # See user head tilting
                if y < -10:
                    text = "Looking left"
                elif y > 10:
                    text = "Looking Right"
                elif x < -10:
                    text = "Looking Down"
                elif x > 10:
                    text = "Looking Up"
                else:
                    text = "Forward"
                # Display nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(
                    nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y*10), int(nose_2d[1] - x*10))
                cv2.line(image, p1, p2, (255, 0, 0), 3)

                cv2.putText(image, text, (0, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.putText(image, "x: " + str(np.round(x, 2)), (0, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "y: " + str(np.round(y, 2)), (0, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "z: " + str(np.round(z, 2)), (0, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            end = time.time()
            totalTime = end-start

            # # fps=1/totalTime
            # fps = totalTime/1
            # print("FPS:", fps)

            # cv2.putText(image, f'FPS:{int(fps)}', (20, 450),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(
                image=image, landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
                
        else:
            text = "not face"
            cv2.putText(image, text, (0, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv2.imshow('Head Pose Estimation', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cap.release()


if __name__ == '__main__':
    app.run(debug=True)
