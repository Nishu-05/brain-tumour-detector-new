from flask import Flask, request, jsonify
from flask import Flask, render_template, redirect, url_for
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import base64
import numpy as np

app = Flask(__name__)
CORS(app)

class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 9 * 9, 512)
        self.relu5 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))
        x = self.flatten(x)
        x = self.dropout(self.relu5(self.fc1(x)))
        x = self.fc2(x)
        return x

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = BrainTumorCNN(num_classes=4)
try:
    model.load_state_dict(torch.load('neuroscan_model.pth', map_location=device))
    model.to(device)
    model.eval() # Set to evaluation mode
    print("✅ Model loaded successfully")
except FileNotFoundError:
    print("❌ Error: 'neuroscan_model.pth' not found. Did you run train_model.py?")

transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
])

CLASS_NAMES = {
    0: 'glioma', 
    1: 'meningioma', 
    2: 'notumor', 
    3: 'pituitary'
}

@app.route("/")
def root():
    return render_template("index.html")
    
@app.route('/predict', methods=['POST'])

def predict():
    data = request.get_json()
    
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    try:
        # Decode Image
        image_data = data['image'].split(",")[1]
        decoded_img = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(decoded_img)).convert('RGB')
        
        # Preprocess
        tensor_image = transform(image).unsqueeze(0)
        tensor_image = tensor_image.to(device)
        
        with torch.no_grad():
            outputs = model(tensor_image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
        class_id = predicted_idx.item()
        confidence_score = confidence.item() * 100
        result_class = CLASS_NAMES.get(class_id, "Unknown")
        
        is_tumor = result_class != 'No Tumor'
        
        return jsonify({
            'tumorDetected': bool(is_tumor),
            'tumorType': result_class,
            'confidence': float(confidence_score),
            # Mapping logic for frontend visualization
            'grade': 4 if 'Glioma' in result_class else (1 if 'meningioma' in result_class else 0),
            'severity': 'High' if 'Glioma' in result_class else 'Low'
        })
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)