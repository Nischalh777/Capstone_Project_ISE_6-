import os
import io
import torch
import torch.nn as nn
import timm  # Add timm for MobileNetV3
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
app.secret_key = os.getenv("FLASK_SECRET_KEY", "a_very_secret_default_key")

# --- Production-Ready Folder Configuration ---
UPLOAD_FOLDER = 'static/uploads'
# Render's persistent disk is at /var/data. We default to a local folder.
DATABASE_HOME = os.getenv("DATABASE_URL", ".") 

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(DATABASE_HOME, "detections.db")}'
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
def load_mobilenetv3_model(model_path, class_names_path):
    """
    Load MobileNetV3 model with proper error handling for different checkpoint formats
    """
    try:
        # Load class names
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
        num_classes = len(class_names)
        print(f"✅ Successfully loaded {num_classes} class names.")
        
        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Using device: {device}")
        
        # Load checkpoint and inspect structure
        print(f"📂 Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                print("✅ Found 'model_state_dict' in checkpoint")
                state_dict = checkpoint['model_state_dict']
                
                # Extract model info from checkpoint if available
                model_name = checkpoint.get('config', {}).get('model_name', 'mobilenetv3_large_100')
                checkpoint_classes = checkpoint.get('num_classes', num_classes)
                
                print(f"📋 Checkpoint info:")
                print(f"   Model: {model_name}")
                print(f"   Classes: {checkpoint_classes}")
                if 'accuracy' in checkpoint:
                    print(f"   Accuracy: {checkpoint['accuracy']:.2f}%")
                    
            elif 'state_dict' in checkpoint:
                print("✅ Found 'state_dict' in checkpoint")
                state_dict = checkpoint['state_dict']
                model_name = 'mobilenetv3_large_100'
                
            else:
                # Assume entire checkpoint is state dict
                print("✅ Treating entire checkpoint as state dict")
                state_dict = checkpoint
                model_name = 'mobilenetv3_large_100'
        else:
            print("❌ Invalid checkpoint format")
            raise ValueError("Checkpoint must be a dictionary")
        
        # Create model architecture
        print(f"🏗️ Creating {model_name} architecture...")
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        
        # Load state dict with error handling
        try:
            model.load_state_dict(state_dict, strict=True)
            print("✅ Model weights loaded successfully (strict mode)")
        except Exception as e:
            print(f"⚠️ Strict loading failed: {e}")
            print("🔄 Trying non-strict loading...")
            model.load_state_dict(state_dict, strict=False)
            print("✅ Model weights loaded successfully (non-strict mode)")
        
        # Move to device and set to eval mode
        model = model.to(device)
        model.eval()
        
        print(f"🎉 MobileNetV3 model loaded successfully!")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Device: {device}")
        
        return model, class_names, device
        
    except FileNotFoundError as e:
        print(f"🔴 FATAL ERROR: File not found - {e}")
        print("Make sure both model file and class_names.json exist in the models/ directory")
        exit()
    except Exception as e:
        print(f"🔴 FATAL ERROR: Failed to load MobileNetV3 model: {e}")
        print("\n🔍 Troubleshooting tips:")
        print("1. Check if the model file is corrupted")
        print("2. Verify the model was trained with the correct number of classes")
        print("3. Ensure the checkpoint format is compatible")
        exit()

# Load model and class names
model_path = 'models/mobilenetv3_large_best.pth'  # Update this to your MobileNetV3 model path
class_names_path = 'models/class_names.json'

model, CLASS_NAMES, device = load_mobilenetv3_model(model_path, class_names_path)
NUM_CLASSES = len(CLASS_NAMES)

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
# Updated transform for MobileNetV3 (same as training)
transform_inference = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_disease(image_bytes):
    """
    Predict disease using MobileNetV3 model
    """
    try:
        # Load and preprocess image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = transform_inference(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_prob, predicted_idx = torch.max(probabilities, 1)
        
        # Get results
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence = predicted_prob.item() * 100
        
        print(f"🎯 Prediction: {predicted_class} ({confidence:.2f}%)")
        return predicted_class, confidence
        
    except Exception as e:
        print(f"🔴 ERROR in predict_disease function: {e}")
        return None, 0.0

def get_chatbot_response(predicted_class):
    """
    Get AI-generated response for the predicted disease
    """
    if 'healthy' in predicted_class.lower():
        crop_name = predicted_class.split('___')[0].replace('_', ' ')
        response_text = f"""### **Great News! 🎉**

The {crop_name} leaf appears to be **healthy**! No signs of disease detected.

### **Maintenance Tips**
- Continue with your current care routine
- Monitor regularly for any changes
- Maintain proper watering and nutrition
- Ensure adequate sunlight and air circulation

### **Prevention**
- Keep the growing area clean
- Avoid overwatering
- Provide proper spacing between plants
- Remove any dead or damaged leaves promptly"""
        return True, response_text
    
    if not gemini_model:
        return False, "Chatbot is not available due to a server configuration issue."
    
    try:
        # Parse disease name
        try:
            crop_name, disease_name = predicted_class.split('___')
            disease_name = disease_name.replace('_', ' ')
            crop_name = crop_name.replace('_', ' ')
        except ValueError:
            crop_name = "Unknown Crop"
            disease_name = predicted_class.replace('_', ' ')
        
        # Create detailed prompt
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
        
        response = gemini_model.generate_content(prompt)
        return True, response.text
        
    except Exception as e:
        print(f"🔴 ERROR calling Gemini API: {e}")
        return False, "Could not connect to the AI chatbot due to a temporary API rate limit. Please wait a minute and try again."

def process_and_save_prediction(image_bytes, original_filename="captured.png"):
    """
    Process image, make prediction, and save to database
    """
    predicted_class, confidence = predict_disease(image_bytes)
    if predicted_class is None:
        return None, None, None
    
    # Save image
    filename = f"{uuid.uuid4()}_{original_filename.replace(' ', '_')}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(filepath, 'wb') as f:
        f.write(image_bytes)
    
    image_url = url_for('static', filename=f'uploads/{filename}')
    
    # Save to database
    with app.app_context():
        new_detection = Detection(
            image_url=image_url, 
            predicted_class=predicted_class, 
            confidence=confidence
        )
        db.session.add(new_detection)
        db.session.commit()
    
    return predicted_class, confidence, image_url

# --- Main Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    detections = Detection.query.order_by(Detection.timestamp.desc()).all()
    return render_template('dashboard.html', detections=detections)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: 
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '': 
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        image_bytes = file.read()
        predicted_class, confidence, image_url = process_and_save_prediction(image_bytes, file.filename)
        
        if predicted_class is None:
            return jsonify({'error': 'Model prediction failed.'}), 500
        
        return jsonify({
            'prediction': predicted_class, 
            'confidence': f"{confidence:.2f}", 
            'image_url': image_url
        })
        
    except Exception as e:
        print(f"🔴 ERROR in /predict route: {e}")
        return jsonify({'error': 'An internal server error occurred'}), 500

@app.route('/capture', methods=['POST'])
def capture():
    try:
        data = request.get_json()
        if 'image' not in data: 
            return jsonify({'error': 'No image data found'}), 400
        
        header, encoded = data['image'].split(",", 1)
        image_bytes = base64.b64decode(encoded)
        predicted_class, confidence, image_url = process_and_save_prediction(image_bytes)
        
        if predicted_class is None:
            return jsonify({'error': 'Model prediction failed.'}), 500
        
        return jsonify({
            'prediction': predicted_class, 
            'confidence': f"{confidence:.2f}", 
            'image_url': image_url
        })
        
    except Exception as e:
        print(f"🔴 ERROR in /capture route: {e}")
        return jsonify({'error': 'An internal server error occurred'}), 500

@app.route('/get_info', methods=['POST'])
def get_info():
    try:
        data = request.get_json()
        if 'prediction' not in data: 
            return jsonify({'error': 'Missing prediction name'}), 400
        
        predicted_class = data['prediction']
        gemini_success, chatbot_response = get_chatbot_response(predicted_class)
        
        return jsonify({
            'chatbot_response': chatbot_response, 
            'gemini_success': gemini_success
        })
        
    except Exception as e:
        print(f"🔴 ERROR in /get_info route: {e}")
        return jsonify({'chatbot_response': "Server error.", 'gemini_success': False}), 500

@app.route('/delete/<int:detection_id>', methods=['POST'])
def delete_detection(detection_id):
    try:
        detection_to_delete = db.session.get(Detection, detection_id)
        if detection_to_delete:
            image_filename = os.path.basename(detection_to_delete.image_url)
            image_filepath = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
            
            db.session.delete(detection_to_delete)
            db.session.commit()
            
            if os.path.exists(image_filepath):
                os.remove(image_filepath)
                
            return jsonify({'success': True, 'message': 'Detection deleted.'})
        else:
            return jsonify({'success': False, 'message': 'Detection not found.'}), 404
            
    except Exception as e:
        print(f"🔴 ERROR in /delete route: {e}")
        db.session.rollback()
        return jsonify({'success': False, 'message': 'An internal server error occurred.'}), 500

@app.route('/setup')
def setup_database():
    """
    This is a one-time setup route. Visiting this URL will create the database tables.
    """
    try:
        print("--- SETUP ROUTE CALLED ---")
        with app.app_context():
            print("Creating database tables...")
            db.create_all()
            print("✅ Database tables should be created now.")
        
        return "Database setup complete! The 'detection' table has been created. You can now use the app.", 200
        
    except Exception as e:
        print(f"🔴 ERROR during setup route: {e}")
        return f"An error occurred during database setup: {e}", 500

# --- Health Check Route (Optional) ---
@app.route('/health')
def health_check():
    """
    Health check endpoint to verify model and system status
    """
    try:
        # Test model prediction with dummy data
        dummy_image = Image.new('RGB', (224, 224), color='red')
        img_byte_arr = io.BytesIO()
        dummy_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        predicted_class, confidence = predict_disease(img_byte_arr)
        
        status = {
            'status': 'healthy',
            'model_loaded': model is not None,
            'device': str(device),
            'num_classes': NUM_CLASSES,
            'gemini_available': gemini_model is not None,
            'test_prediction': predicted_class is not None
        }
        
        return jsonify(status), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')