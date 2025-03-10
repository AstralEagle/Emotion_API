from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import base64
from io import BytesIO

db_uri = ""

# client = MongoClient(db_uri, server_api=ServerApi('1'))
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}



model_name = "trpakov/vit-face-expression"
model = AutoModelForImageClassification.from_pretrained(model_name)
extractor = AutoFeatureExtractor.from_pretrained(model_name)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def hello_world():  # put application's code here
    try:
        # client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
    return 'Hello World!'


@app.route("/upload", methods=["POST"])
def upload_api():
    # if request.method == "OPTIONS":
        # return _build_cors_prelight_response()
    
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Aucune image Base64 reçue"}), 400

        base64_image = data["image"]
        
        if "," in base64_image:
            base64_image = base64_image.split(",")[1]

        image_bytes = base64.b64decode(base64_image)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        inputs = extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1)

        emotion_predite = model.config.id2label[predicted_class.item()]
        print(f"Émotion prédite: {emotion_predite}")

        print(emotion_predite)

        return _corsify_actual_response(jsonify({
            "message": "Image traitée avec succès",
            "emotion": emotion_predite
        }))

    except Exception as e:
        return _corsify_actual_response(jsonify({"error": str(e)}), 500)

def _build_cors_prelight_response():
    response = jsonify({"message": "CORS preflight successful"})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


if __name__ == '__main__':
    app.run()
