from flask import Flask, request, jsonify
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from werkzeug.utils import secure_filename
from PIL import Image
import torch

db_uri = ""

# client = MongoClient(db_uri, server_api=ServerApi('1'))
app = Flask(__name__)
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
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier envoyé"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Fichier vide"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        image = Image.open(file_path).convert("RGB")

        inputs = extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1)

        os.remove(file_path)

        # Afficher le résultat
        print(f"Émotion prédite: {model.config.id2label[predicted_class.item()]}")
        return jsonify({"message": "Fichier uploadé avec succès", "path": file_path, "emotion": model.config.id2label[predicted_class.item()]})
    return jsonify({"error": "Format de fichier non autorisé"}), 400



if __name__ == '__main__':
    app.run()
