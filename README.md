# French Sign Language Detection and Conversion
This project aims to detect and convert French Sign Language (LSF) gestures into text using computer vision techniques. The application utilizes the Mediapipe library for hand and pose detection, and a trained LSTM (Long Short-Term Memory) model for gesture classification.

**Features**
Real-time detection and conversion of French Sign Language gestures into text.
Visualization of hand and pose landmarks on captured video feed.
Training and fine-tuning of the gesture recognition model.
Easy-to-use interface for capturing and processing video data.


**Installation**
Clone the repository:
```
[git clone https://github.com/Khaldi-o/LSF_AI.git
```
Install the required dependencies:
```
pip install -r requirements.txt
```
Download the pre-trained model weights or train your own model using the provided scripts.
Run the main application script:
```
cd action_recognition
python functions.py
```
Point your camera towards the signer's hand gestures.
The detected gestures will be converted into text and displayed on the screen in real-time.
Training
If you want to train your own gesture recognition model, follow these steps:

- Prepare your dataset of sign language gestures.
- Run the data collection script to capture and save hand and pose landmarks.
- Process the collected data and split it into training and testing sets.
- Train the LSTM model using the provided training script.
- Evaluate the trained model's performance and fine-tune as necessary.
