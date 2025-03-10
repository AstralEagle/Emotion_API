from flask import Flask, request, jsonify
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from mongoengine import Document, StringField, ListField, ReferenceField, DateTimeField, EmbeddedDocument, EmbeddedDocumentField, CASCADE, connect
import datetime
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity
import jwt


db_uri = "mongodb+srv://arthurdias:fL4OVc5V1jl85rMn@cluster0.ff58v.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"


connect('mydatabase', host=db_uri)

class Tendance(EmbeddedDocument):
    tendance = StringField(required=True)  # Exemple : 'positive', 'negative', etc.
    value = StringField(required=True)

class User(Document):
    username = StringField(required=True, unique=True)
    email = StringField(required=True, unique=True)
    password = StringField(required=True)
    bio = StringField(default="")
    avatar = StringField(default="")
    followers = ListField(ReferenceField('User', reverse_delete_rule=CASCADE))
    following = ListField(ReferenceField('User', reverse_delete_rule=CASCADE))
    createdAt = DateTimeField(default=datetime.datetime.utcnow)
    tendances = ListField(EmbeddedDocumentField(Tendance), default=[])

client = MongoClient(db_uri, server_api=ServerApi('1'))
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

app.config['JWT_SECRET_KEY'] = "secret"

jwt = JWTManager(app)




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
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
    return 'Hello World!'

@app.route('/protected', methods=['GET'])
@jwt_required()  # Le décorateur jwt_required vérifie la validité du token
def protected():
    print("Access granted!")
    # Obtenez l'identité de l'utilisateur (dans ce cas, le "username" du JWT)
    current_user = get_jwt_identity()
    return jsonify(logged_in_as=current_user), 200


@app.route('/validate-token', methods=['POST'])
def validate_token():
    token = request.json.get('token', None)
    if not token:
        return jsonify({"msg": "Token manquant"}), 400

    try:
        # Utiliser pyjwt pour décoder le token manuellement avec le même secret que dans l'API Express
        decoded_token = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=["HS256"])
        return jsonify({"msg": "Token valide", "decoded_token": decoded_token}), 200
    except jwt.ExpiredSignatureError:
        return jsonify({"msg": "Le token a expiré"}), 401
    except jwt.InvalidTokenError:
        return jsonify({"msg": "Token invalide"}), 401

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
