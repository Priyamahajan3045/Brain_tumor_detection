⚙️ Installation

Clone the repository and install dependencies:

git clone https://github.com/your-username/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
pip install -r requirements.txt

📊 Model Details
Base Model: ResNet50 (pre-trained on ImageNet)
Input Shape: 224 x 224 x 3
Layers Added:
GlobalAveragePooling2D
Dense (256, ReLU)
Dropout (0.5)
Dense (1, Sigmoid)
Loss Function: Binary Crossentropy
Optimizer: Adam
🚀 Usage
1. Train the Model
history = model.fit(train_data, validation_data=val_data, epochs=10)

2. Save the Model
model.save("brain_tumor_model.h5")

3. Load & Predict
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = load_model("brain_tumor_model.h5")

# Load and preprocess image
img = image.load_img("sample.jpg", target_size=(224,224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Prediction
prediction = model.predict(img_array)
if prediction[0][0] > 0.5:
    print("🧠 Brain Tumor Detected")
else:
    print("✅ No Brain Tumor")

📷 Sample Predictions

✅ No Brain Tumor

🧠 Brain Tumor Detected

✨ Future Improvements

Add data augmentation for better generalization

Deploy as Streamlit / Flask Web App

Use Grad-CAM for visual tumor localization
