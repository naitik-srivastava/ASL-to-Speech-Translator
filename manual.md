             ## ASL Translator Manual

Created by: Naitik Srivastava
Model Used: models/best.pt (YOLOv8n-classifier trained on 20k+ ASL images)
Language Recognition: 29 Classes (A–Z, space, del, nothing)

✅ Prerequisites
Before running the translator, make sure you’ve installed all dependencies. You can do this with:

bash
Copy
Edit
pip install -r requirements.txt
📁 Folder Structure
bash
Copy
Edit
project_root/
├── models/
│   └── best.pt
│   └── images    
├── main_Asl_To_Speech.py # This script (main file)
├── requirements.txt
├── manual.md               # This file
├── models_readme.txt       # Model details
▶️ How to Run
Ensure your webcam is connected and functional.

Navigate to the project directory in your terminal.

Run the script:

bash
Copy
Edit
python asl_translator.py
✋ How It Works
Left Hand = Input Gesture

Right Hand = Register Gesture

The system uses:

YOLOv8 classifier for gesture prediction (left hand)

Right-hand presence to trigger word registration

Face mesh and hand landmarks for visual feedback

Real-time video through OpenCV

Sentence construction and text-to-speech output

⌨️ Controls
Show translation as text: Automatic (top of video feed)

Speak the sentence aloud: Press T

Quit the application: Press Q

🧠 Gesture Logic
Gesture Class	Function
A–Z	Appends letter
space	Appends a space
del	Deletes last letter
nothing	Ignores input

Right hand must be visible in the frame to register a left-hand gesture.

If no gesture is detected, the system waits silently.

Sentence is built incrementally and can be spoken using T.

🗣️ Voice Output
Text-to-speech is powered by gTTS and pygame.

When you press T, the sentence is converted to audio and played.

Temporary MP3 files are cleaned up automatically.

💡 Tips for Best Performance
Ensure good lighting and a clean background.

Keep hands centered in the frame.

Avoid rapid hand switching between left and right.

Maintain hand gestures for at least a second for stable recognition.

Don't overlap hands to avoid misclassification.

Advanced Users
You can modify the cooldown time (cooldown = 0.8) to control registration frequency.

The model expects 224x224 input; changing this will require re-training or resizing logic.

You can log predictions by inserting print(class_name, confidence) inside the loop.

 
 
 
              License & Credits
© 2025 Naitik Srivastava. All rights reserved.
Use permitted for personal and academic purposes with credit.
Commercial use requires permission.