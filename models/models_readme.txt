==========================================================
YOLOv8 Hand Gesture Recognition Model for ASL (29 Classes)
==========================================================

Author:           Naitik Srivastava  
Model File:       best.pt  
Framework:        Ultralytics YOLOv8  
Model Type:       Image Classification  
Date Trained:     July 22, 2025  
----------------------------------------------------------

🔍 Overview:
------------
This model was specifically trained to classify American Sign Language (ASL) hand gestures using YOLOv8's image classification capabilities. It supports 29 distinct classes, which include:

- 26 Alphabet gestures: A–Z  
- 3 functional commands: `space`, `del`, and `nothing`  

The model is the result of training on a diverse and robust dataset of over 20,000+ labeled images, ensuring high accuracy, consistency, and real-time usability in interactive ASL translation applications.
In the provided main_Asl_To_Speech.py you can use it to convert it from text to speech output using gTTS.

----------------------------------------------------------

📊 Dataset Summary:
-------------------
- Total Images:         ~20,000+
- Number of Classes:    29  
- Images per Class:     
  • ~540+ training images per class  
  • ~150 validation images per class

- Classes List:  
  A, B, C, D, E, F, G, H, I, J, K, L, M,  
  N, O, P, Q, R, S, T, U, V, W, X, Y, Z,  
  space, del, nothing

- Data Split:  
  • 80% Training(approx)
  • 20% Validation(approx)

- Image Size: 224x224 pixels  
- Image Type: RGB (jpg format)  
- Annotations: Single-label image classification only

- Preprocessing & Augmentations:  
  ✓ Resizing to 224x224  
  ✓ Random horizontal flips  
  ✓ Rotation (±10–15°)  
  ✓ Brightness/contrast variations  
  ✓ Standard normalization

----------------------------------------------------------

Training Configuration:
--------------------------
- Model Architecture:      YOLOv8n-cls (Nano Classifier)  
- Epochs:                  35  
- Batch Size:              32  
- Learning Rate:           0.0001  
- Optimizer:               Adam  
- Training Time:           ~7 hours  
- Device Used:             Trained Without GPU just on a 4gb ram Laptop goddammit  
- Framework Version:       Ultralytics 8.x (July 2025)

- Training Platform:  
  Local training using Python environment with Ultralytics and OpenCV integration.

----------------------------------------------------------

🏆 Performance Metrics:
-----------------------
- Final Training Loss:     < 0.09  
- Top-1 Accuracy:          99.9%  
- Top-5 Accuracy:          100%  
- Overfitting:             None observed  
- Validation Stability:    High consistency across all classes

This model has been tested in real-time ASL translation scenarios with excellent responsiveness and stability. The 0.999 Top-1 accuracy ensures nearly perfect single-shot predictions, and Top-5 accuracy of 1.0 confirms the model's robustness even with slight hand variations.

----------------------------------------------------------

💡 Integration & Usage:
-----------------------
You can integrate this model in any Python project using the Ultralytics API. It has been successfully integrated into a real-time ASL translator that detects the left-hand gesture and waits for the right-hand signal to register the word.

Basic Usage:

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO("models/best.pt")

# Read and preprocess image (resize to 224x224 if not done)
img = cv2.imread("sample.jpg")
img = cv2.resize(img, (224, 224))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Get prediction
results = model(img_rgb)
predicted_class = results[0].names[results[0].probs.top1]
confidence = results[0].probs.top1conf

print(f"Prediction: {predicted_class}, Confidence: {confidence:.2f}")


© 2025 Naitik Srivastava. This model is licensed for personal and academic use only. Commercial use requires explicit permission.
