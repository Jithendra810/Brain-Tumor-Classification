from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "best_Xception.keras"  # Make sure this path is correct
model = load_model(MODEL_PATH)

class_labels = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/classification", methods=["GET", "POST"])
def classification():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        # Save the file
        file_path = os.path.join("static/uploads", file.filename)
        file.save(file_path)

        # Preprocess image (resize, normalize)
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

        # Make a prediction
        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        return render_template("classification.html", 
                               filename=file.filename, 
                               prediction=predicted_class, 
                               confidence=confidence)

    return render_template("classification.html")

if __name__ == "__main__":
    # Create the uploads directory if it doesn't exist
    if not os.path.exists("static/uploads"):
        os.makedirs("static/uploads")
    
    app.run(debug=True)