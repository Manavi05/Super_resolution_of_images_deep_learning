from flask import Flask, request, render_template, send_from_directory
import os
import numpy as np
from PIL import Image
from datetime import datetime
from load_realesrgan_model import load_realesrgan_model
import cv2
import matplotlib.pyplot as plt

# LIME + Classifier
from lime import lime_image
from skimage.segmentation import mark_boundaries
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Load models
upsampler = load_realesrgan_model('models/RealESRGAN_x4plus.pth')
classifier_model = mobilenet_v2.MobileNetV2(weights='imagenet')

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def preprocess_image(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((128, 128))  # Simulate low-res
    img = np.array(img).astype(np.float32) / 255.0
    return img

def super_resolve_image(img_array):
    img_uint8 = (img_array * 255).astype(np.uint8)
    output, _ = upsampler.enhance(img_uint8)
    return np.array(output).astype(np.float32) / 255.0

def predict_fn(images):
    images_resized = np.array([cv2.resize(img, (224, 224)) for img in images])
    images_preprocessed = preprocess_input(images_resized.copy())
    preds = classifier_model.predict(images_preprocessed)
    return preds

def explain_lime_on_image(img_array, save_path):
    explainer = lime_image.LimeImageExplainer()
    
    explanation = explainer.explain_instance(
        img_array,
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )
    
    temp, mask = explanation.get_image_and_mask(
        label=explanation.top_labels[0],
        positive_only=True,
        hide_rest=False,
        num_features=10,
        min_weight=0.0
    )

    lime_result = mark_boundaries(temp / 255.0, mask)
    plt.imsave(save_path, lime_result)

@app.route("/", methods=["GET", "POST"])
def index():
    gallery_files = sorted([
        f for f in os.listdir(RESULT_FOLDER)
        if f.startswith("sr_") and f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ], key=lambda x: os.path.getmtime(os.path.join(RESULT_FOLDER, x)), reverse=True)[:8]

    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = os.path.splitext(image_file.filename)[0]
            unique_name = f"{base_filename}_{timestamp}.png"
            image_path = os.path.join(UPLOAD_FOLDER, unique_name)
            result_filename = f"sr_{unique_name}"
            result_path = os.path.join(RESULT_FOLDER, result_filename)
            lime_path = os.path.join(RESULT_FOLDER, f"lime_{unique_name}")

            image_file.save(image_path)
            img = preprocess_image(image_path)
            sr_img = super_resolve_image(img)
            Image.fromarray((sr_img * 255).astype(np.uint8)).save(result_path)

            explain_lime_on_image((sr_img * 255).astype(np.uint8), lime_path)

            return render_template("result.html",
                                   input_image_path=f"uploads/{unique_name}",
                                   output_image_path=f"results/{result_filename}",
                                   lime_image_path=f"results/lime_{unique_name}",
                                   output_image_name=result_filename)

    return render_template("index.html", gallery_images=gallery_files)

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory(RESULT_FOLDER, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
