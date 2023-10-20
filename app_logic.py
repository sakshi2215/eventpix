import os
from image_processing import process_image
from tensorflow.keras.models import load_model
import shutil

# Update the folder path to the correct path
input_folder_path = os.path.join(os.path.dirname(__file__), 'images')

def segregate_images(model):
    output_base = 'output'
    os.makedirs(os.path.join(output_base, 'closed_eyes'), exist_ok=True)
    os.makedirs(os.path.join(output_base, 'open_eyes'), exist_ok=True)

    for file in os.listdir(input_folder_path):
        file_path = os.path.join(input_folder_path, file)
        if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg')):
            result = process_image(file_path, model)
            if result is not None:
                output_folder = os.path.join(output_base, 'closed_eyes' if result == 'Closed Eyes' else 'open_eyes')
                output_file_path = os.path.join(output_folder, file)

                # Copy the file to the output folder instead of moving it
                shutil.copy(file_path, output_file_path)

# Usage
if __name__ == '__main__':
    model = load_model('models\my_model.h5')  # Load your model
    segregate_images(model)
