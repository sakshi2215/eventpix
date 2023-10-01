from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory, jsonify
import os
import shutil
from emotion_detection import predict_emotion  # Import your emotion prediction function
from tensorflow.keras.models import load_model
from image_processing import process_image
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import io
from PIL import Image
import base64
from Helpers import *
import hashlib
import shutil
from tkinter import filedialog
import tkinter as tk
import cv2
import mediapipe as mp

from PIL import Image, ImageFilter
import pilgram,pilgram.css
from filter import apply_filter_to_whole_image ,apply_filter_to_face


app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your machine learning model
model = load_model('models\my_model.h5')
mp_face_detection = mp.solutions.face_detection
# Define the allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_photo():
    images_outputs = []
    if request.method == 'POST':
        files = request.files.getlist('files[]')

        for file in files:
            if file and allowed_file(file.filename):
                file_path = os.path.join('uploads', file.filename)
                file.save(file_path)
                result = process_image(file_path, model)
                 # Create dictionary for image and output
                image_output = {
                    'image_path': file_path,
                    'output': result if result else 'Unable to determine eye status'
                }
                images_outputs.append(image_output)

                if result:
                    output_folder = os.path.join('output', 'closed_eyes' if result == 'Closed Eyes' else 'open_eyes')
                    os.makedirs(output_folder, exist_ok=True)

                    # Move the file to the appropriate output folder
                    output_file_path = os.path.join(output_folder, file.filename)
                    shutil.move(file_path, output_file_path)

                    flash('Eye status for ' + file.filename + ': ' + result, 'success')
                else:
                    flash('Unable to determine eye status for ' + file.filename, 'warning')

        return render_template('index.html', image_output=image_output)

    return render_template('index.html')

@app.route('/emotion_prediction.html')
def emotion_prediction_page():
    return render_template('emotion_prediction.html')

@app.route('/detect_emotion', methods=['GET', 'POST'])
def detect_emotion():
    files = request.files.getlist('emotion_file')
    for file in files:
        if file and allowed_file(file.filename):
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

        # Perform emotion prediction
        predicted_emotion = predict_emotion(file_path)

        if predicted_emotion:
            flash('Predicted emotion for ' + file.filename + ': ' + predicted_emotion, 'success')
        else:
            flash('Unable to determine emotion for ' + file.filename, 'warning')

    return render_template('emotion_prediction.html')

@app.route('/blur_detection.html')
def blur_prediction_page():
    return render_template('blur_detection.html')


@app.route('/detect_blur', methods=['POST'])
def upload_image():
	images = []
	for file in request.files.getlist("blur_file[]"):
		print("***************************")
		print("image: ", file)
		if file.filename == '':
			flash('No image selected for uploading')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			filestr = file.read()
			npimg = np.frombuffer(filestr, np.uint8)
			image = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
			ratio = image.shape[0] / 500.0
			orig = image.copy()
			image = Helpers.resize(image, height = 500)

			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			fm = cv2.Laplacian(gray, cv2.CV_64F).var()
			result = "Not Blurry"

			if fm < 100:
				result = "Blurry"

			sharpness_value = "{:.0f}".format(fm)
			message = [result,sharpness_value]

			img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			file_object = io.BytesIO()
			img= Image.fromarray(Helpers.resize(img,width=500))
			img.save(file_object, 'PNG')
			base64img = "data:image/png;base64,"+base64.b64encode(file_object.getvalue()).decode('ascii')
			images.append([message,base64img])

	print("images:", len(images))
	return render_template('blur_detection.html', images=images )

@app.route('/duplicate_detection.html')
def duplicate_prediction_page():
    return render_template('duplicate_detection.html')

@app.route('/duplicate_detection', methods=['POST'])
def duplicate_detection():
    root = tk.Tk()
    root.withdraw()
    
    # Ask the user to select a directory
    directory = filedialog.askdirectory()
    
    if not directory:
        flash('No directory selected. Please try again.', 'danger')
        return redirect(request.url)
    
    # Function to get hash for a file
    def get_hash(file_path):
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as file:
            while chunk := file.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    # Function to scan the directory for images
    def scan_directory(directory):
        images = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(('.png', '.jpg')):
                    images.append(os.path.join(root, file))
        return images
    
    # Function to find duplicates and move them to a "duplicates" folder
    def find_duplicates(directory):
        hashes = {}
        duplicates = []
        images = scan_directory(directory)
        for image in images:
            image_hash = get_hash(image)
            if image_hash in hashes:
                duplicates.append(image)
            else:
                hashes[image_hash] = image
        duplicate_folder = os.path.join(directory, 'duplicates')
        if not os.path.exists(duplicate_folder):
            os.makedirs(duplicate_folder)
        for duplicate in duplicates:
            shutil.move(duplicate, os.path.join(duplicate_folder, os.path.basename(duplicate)))
        return duplicates
    
    # Perform duplicate detection and get duplicate images
    duplicate_images = find_duplicates(directory)
    duplicate_image_paths = [os.path.join('duplicates', os.path.basename(image)) for image in duplicate_images]
    
    # Return duplicate images as JSON response
    return jsonify(duplicate_images=duplicate_image_paths)
@app.route('/filter.html')
def filter_page():
    return render_template('filter.html')

@app.route('/filter_detection', methods=['POST'])
def uploadfilter_image():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Save the uploaded file
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

        # Get filter choices from the form
        filter_type_whole_image = request.form.get('filter_type_whole_image')
        filter_type_face = request.form.get('filter_type_face')

        # Load the image using OpenCV for face detection
        cv_image = cv2.imread(image_path)

        # Apply filters to the whole image and the face
        image = Image.open(image_path)
        image_whole_image_filtered = apply_filter_to_whole_image(image.copy(), filter_type_whole_image)
        image_face_filtered = apply_filter_to_face(cv_image.copy(), filter_type_face)

        # Save the filtered images
        filtered_image_path_whole_image = os.path.join(app.config['UPLOAD_FOLDER'], 'filtered_whole_image_' + os.path.basename(image_path))
        filtered_image_path_face = os.path.join(app.config['UPLOAD_FOLDER'], 'filtered_face_' + os.path.basename(image_path))
        image_whole_image_filtered.save(filtered_image_path_whole_image)
        cv2.imwrite(filtered_image_path_face, image_face_filtered)

        return render_template('result.html', original_image=file.filename,
                               filtered_image_whole_image=os.path.basename(filtered_image_path_whole_image),
                               filtered_image_face=os.path.basename(filtered_image_path_face))
    else:
        return render_template('error.html')
    
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

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
