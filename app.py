import os
import io
import torch
import torch.nn as nn
from torchvision.models import resnet50
from flask import Flask, render_template, request, jsonify, url_for
from dotenv import load_dotenv
import google.generativeai as genai
import uuid
import json
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
import base64
from torchvision import transforms
from PIL import Image

# --- App Initialization and Configuration ---
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "a_default_secret_key")
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///detections.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Database Model ---
class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    image_url = db.Column(db.String(100), nullable=False)
    predicted_class = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)

# --- Global Model Loading ---
try:
    with open('models/class_names.json', 'r') as f:
        CLASS_NAMES = json.load(f)
    NUM_CLASSES = len(CLASS_NAMES)
    print(f"✅ Successfully loaded {NUM_CLASSES} class names.")
except FileNotFoundError:
    print("🔴 FATAL ERROR: 'models/class_names.json' not found. Make sure it's in the 'models' folder.")
    exit()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'models/plant_disease_model.pth'
model = resnet50(weights=None)
model.fc = nn.Linear(in_features=2048, out_features=NUM_CLASSES)
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ PyTorch model loaded successfully.")
except Exception as e:
    print(f"🔴 FATAL ERROR: Failed to load PyTorch model: {e}")
    exit()

# --- Gemini API Setup ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash-lite')
        print("✅ Gemini API configured.")
    except Exception as e:
        print(f"⚠️ WARNING: Could not configure Gemini API: {e}")
else:
    print("⚠️ WARNING: GEMINI_API_KEY not set in .env file.")

# --- Helper Functions ---
transform_inference = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_disease(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = transform_inference(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_prob, predicted_idx = torch.max(probabilities, 1)
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence = predicted_prob.item() * 100
        return predicted_class, confidence
    except Exception as e:
        print(f"🔴 ERROR in predict_disease function: {e}")
        return None, 0.0

def get_chatbot_response(predicted_class):
    if 'healthy' in predicted_class.lower():
        crop_name = predicted_class.split('___')[0].replace('_', ' ')
        response_text = f"The {crop_name} leaf appears to be healthy. No specific disease management is required. Continue good practices like proper watering, fertilization, and monitoring."
        return True, response_text
    if not gemini_model:
        return False, "Chatbot is not available due to a server configuration issue."
    try:
        crop_name, disease_name = predicted_class.split('___')
        disease_name = disease_name.replace('_', ' ')
        crop_name = crop_name.replace('_', ' ')
    except ValueError:
        crop_name = "Unknown Crop"
        disease_name = predicted_class.replace('_', ' ')
    prompt = (f"You are an expert plant pathologist advising a farmer. Provide clear and actionable advice for the disease '{disease_name}' affecting a '{crop_name}' plant. "
              "Structure your response with the following Markdown headings:\n\n"
              "### **Description**\n"
              "(Provide a brief, easy-to-understand description of the disease.)\n\n"
              "### **Symptoms**\n"
              "(List the key visual symptoms a farmer should look for.)\n\n"
              "### **Management and Treatment**\n"
              "(Provide a few practical, step-by-step management and treatment strategies.)\n\n"
              "### **Prevention**\n"
              "(List preventative measures to reduce the risk of future infections.)")
    try:
        response = gemini_model.generate_content(prompt)
        return True, response.text
    except Exception as e:
        print(f"🔴 ERROR calling Gemini API: {e}")
        return False, "Could not connect to the AI chatbot due to a temporary API rate limit. Please wait a minute and try again."

# --- Main Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    detections = Detection.query.order_by(Detection.timestamp.desc()).all()
    return render_template('dashboard.html', detections=detections)

def process_and_save_prediction(image_bytes, original_filename="captured.png"):
    predicted_class, confidence = predict_disease(image_bytes)
    if predicted_class is None:
        return None, None, None
    filename = f"{uuid.uuid4()}_{original_filename.replace(' ', '_')}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(filepath, 'wb') as f:
        f.write(image_bytes)
    image_url_for_db = url_for('static', filename=f'uploads/{filename}')
    new_detection = Detection(image_url=image_url_for_db, predicted_class=predicted_class, confidence=confidence)
    db.session.add(new_detection)
    db.session.commit()
    return predicted_class, confidence, image_url_for_db

# In app.py

# ... (after the @app.route('/dashboard') function) ...

@app.route('/delete/<int:detection_id>', methods=['POST'])
def delete_detection(detection_id):
    """
    Deletes a specific detection record from the database.
    """
    try:
        # Find the detection by its ID
        detection_to_delete = Detection.query.get(detection_id)
        
        if detection_to_delete:
            # If found, delete the associated image file from the server
            if os.path.exists(detection_to_delete.image_url):
                 os.remove(detection_to_delete.image_url)

            # Then, delete the record from the database
            db.session.delete(detection_to_delete)
            db.session.commit()
            
            # Return a success message
            return jsonify({'success': True, 'message': 'Detection deleted successfully.'})
        else:
            # If no detection with that ID was found
            return jsonify({'success': False, 'message': 'Detection not found.'}), 404
            
    except Exception as e:
        print(f"🔴 ERROR in /delete route: {e}")
        db.session.rollback() # Rollback the session in case of an error
        return jsonify({'success': False, 'message': 'An internal server error occurred.'}), 500

# ... (the rest of your routes like /predict, /capture, etc.)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No file selected'}), 400
    try:
        image_bytes = file.read()
        predicted_class, confidence, image_url = process_and_save_prediction(image_bytes, file.filename)
        if predicted_class is None:
            return jsonify({'error': 'Model prediction failed.'}), 500
        return jsonify({'prediction': predicted_class, 'confidence': f"{confidence:.2f}", 'image_url': image_url})
    except Exception as e:
        print(f"🔴 ERROR in /predict route: {e}")
        return jsonify({'error': 'An internal server error occurred'}), 500

@app.route('/capture', methods=['POST'])
def capture():
    try:
        data = request.get_json()
        if 'image' not in data: return jsonify({'error': 'No image data found'}), 400
        header, encoded = data['image'].split(",", 1)
        image_bytes = base64.b64decode(encoded)
        predicted_class, confidence, image_url = process_and_save_prediction(image_bytes)
        if predicted_class is None:
            return jsonify({'error': 'Model prediction failed.'}), 500
        return jsonify({'prediction': predicted_class, 'confidence': f"{confidence:.2f}", 'image_url': image_url})
    except Exception as e:
        print(f"🔴 ERROR in /capture route: {e}")
        return jsonify({'error': 'An internal server error occurred'}), 500

@app.route('/get_info', methods=['POST'])
def get_info():
    try:
        data = request.get_json()
        if 'prediction' not in data: return jsonify({'error': 'Missing prediction name'}), 400
        predicted_class = data['prediction']
        gemini_success, chatbot_response = get_chatbot_response(predicted_class)
        return jsonify({'chatbot_response': chatbot_response, 'gemini_success': gemini_success})
    except Exception as e:
        print(f"🔴 ERROR in /get_info route: {e}")
        return jsonify({'chatbot_response': "Server error.", 'gemini_success': False}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
app.run(debug=True, host='0.0.0.0')
