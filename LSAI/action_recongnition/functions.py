#%%
import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt 
import time
import mediapipe as mp

import keras
from keras import *

from sklearn.model_selection import train_test_split
import tensorflow
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


# Set up Mediapipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    """
    Perform detection using Mediapipe Holistic model.

    Args:
        image (numpy.ndarray): Input image.
        model: Mediapipe Holistic model.

    Returns:
        image (numpy.ndarray): Processed image.
        results: Detection results.
    """
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    """
    Draw styled landmarks on the image.

    Args:
        image (numpy.ndarray): Input image.
        results: Detection results.
    """
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
        mp_drawing.DrawingSpec(color=(80,22,10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    )

    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
    )

    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121,22,76), thickness= 2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121,44,230), thickness=2, circle_radius=2)
    )

    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2),
    )

def test_styled_capture():
    """
    Test capturing video with styled landmarks.
    """
    with hol as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            cv.imshow('OpenCV Feed', image)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()

def extract_keypoints(results):
    """
    Extract keypoints from detection results.

    Args:
        results: Detection results.

    Returns:
        numpy.ndarray: Extracted keypoints.
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def create_input_data_folders():
    """
    Create folders for storing input data.
    """
    for action in actions:
        for sequence in range(no_sequences):
            try: 
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

def collect_keypoints_values():
    """
    Collect and save keypoints values.
    """
    cap = cv.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as holistic:
        for action in actions:
            for sequence in range(no_sequences):
                for num_frame in range(sequence_length):
                    ret, frame = cap.read()
                    if ret == True:
                        image, results = mediapipe_detection(frame, holistic)
                        draw_styled_landmarks(image, results)
                        cv.imshow('Collection of Data', image)
                        keypoints = extract_keypoints(results)
                        npy_path = os.path.join(DATA_PATH, action, str(sequence), str(num_frame))
                        np.save(npy_path, keypoints)
                    if cv.waitKey(10) & 0xFF == ord('q'):
                        break
    cap.release()
    cv.destroyAllWindows()

def process_data():
    """
    Process collected data.
    """
    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for num_frame in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(num_frame)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
    return sequences, labels

def split_data(sequences, labels):
    """
    Split data into train and test sets.

    Args:
        sequences: Input sequences.
        labels: Labels for sequences.

    Returns:
        Train and test data splits.
    """
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test

def optimizer(params):
    """
    Define optimizer.

    Args:
        params: Learning rate.

    Returns:
        Optimizer object.
    """
    return keras.optimizers.Adam(learning_rate=params)

def monitor_training():
    """
    Set up Tensorboard for monitoring training.
    """
    log_dir = os.path.join('models', 'logs')
    tb_callback = TensorBoard(log_dir=log_dir)
    return log_dir, tb_callback

def build_model():
    """
    Build LSTM model.

    Returns:
        LSTM model.
    """
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    return model

def train_model(model, params, X_train, y_train):
    """
    Train the LSTM model.

    Args:
        model: LSTM model.
        params: Learning rate.
        X_train: Training data.
        y_train: Training labels.
    """
    optim = optimizer(params)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    log_dir, tb_callback = monitor_training()
    model.fit(X_train, y_train, epochs=num_epochs, callbacks=[tb_callback])
    model.summary()
    model.save(model_pth)

def test_model():
    """
    Test the trained model in real-time.
    """
    model = keras.models.load_model(model_pth)
    sequence = []
    sentence = []
    threshold = 0.5
    cap = cv.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            image , results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            keypoints = extract_keypoints(results)
            sequence.insert(0, keypoints)
            sequence = sequence[:sequence_length]
            if len(sequence) == sequence_length:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])
                if len(sentence) > 5:
                    sentence = sentence[-5:]
            cv.rectangle(image, (0,0), (640,40), (245, 117, 16), -1)
            cv.putText(image, ' '.join(sentence), (3,30),
                        cv.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 2, cv.LINE_AA)
            cv.imshow('Analysing', image)
            if cv.waitKey(100) & 0xFF == ord('q'):
                break
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    test_model()


    #################### Test Styled Capture

    test_styled_capture()

    #################### End Styled Capture

    #################### Create Input Data Folders

    # create_input_data_folders()

    #################### End of Creation 

    #################### Start Collecting & Saving Keypoints Values 

    # collecte_keypoints_values()

    #################### End Collecting & Saving Keypoints Values

    #################### Start Processing Collected Data

    # process_data()

    #################### End Processing Collected DATA

    #################### Start Spliting Collected DATA into train_df & test_df

    # buil_model()

    #################### End Spliting Collected DATA into train_df & test_df
    
    #################### Start Building & Training

    # sequences, labels = process_data()
    # X_train, X_test, y_train, y_test = split_data(sequences, labels)
    # model = buil_model()
    # params = 0.7
    # train_model(model, params, X_train, y_train)

    ##################### End Building & Training

    #################### Start testing Weights

    test_model()

    #################### End testing Weights
# %%
