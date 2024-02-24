#%%

#################### Importing and Installing needed Dependencied and Libriries
import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt 
import time
import mediapipe as mp
# from params import *

import keras
from keras import *

from sklearn.model_selection import train_test_split
import tensorflow
from keras.utils import to_categorical
from keras.models import load_model


#from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

############ Import the Params and Variables file
from params import *
####################################################
print('All needed Dependencies are loaded and installed')

#%%
#################### Extracting Keypoints using MP Holistic

print('Importing MP Holistic Model ....')
mp_holistic = mp.solutions.holistic     # Holistic Model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
print('MP holistic Model Loaded successfuly')


def mediapipe_detection(image, model):

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)    # Color Conversion BGR to RGB
    image.flags.writeable = False                   # Image is no longer writeable
    results = model.process(image)                  # Make prediction
    image.flags.writeable = True                    #Image is now writeable
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)    # Color conversion RGB to BGR
    print('resultes of mediapipe detection :', results)
    return image, results


print('mediapipe_detection() is Ok to run')


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)      # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)          # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)     # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)    # Draw Right hand connections
print('draw_landmarks() is Ok to run')


def draw_styled_landmarks(image, results):
    #Draw face connections
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
        mp_drawing.DrawingSpec(color=(80,22,10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    )


    # Draw pose connections
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
    )

    # Draw Left Hand connections 
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121,22,76), thickness= 2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121,44,230), thickness=2, circle_radius=2)
    )

    #Draw Right Hand Connections
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2),
    )
print('draw_styled_landmarks() is Ok to run')


# result_test = extract_keypoints(results)

#%%
#################### Applying on camera

print('Opening the Camera with styled_landmarks')
def test_styled_capture():

    with hol as holistic :
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            # print(results)

            # Draw Landmarks
            draw_styled_landmarks(image, results)

            # Show to screen
            cv.imshow('OpenCV Feed', image)

            # Break gracefully 
            if cv.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()

    draw_landmarks(frame, results)
    plt.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
print('the Camera is Opened successfuly')

#%%
#################### Extract keypoints value 

# Extract keypoints

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])
print('Extracting keypoints.....')
#################### Setup Folders for collection 


# %%
# Create the extracted input data folder 

def create_input_data_folders():
    for action in actions: 
        for sequence in range(no_sequences):
            try: 
                os.makedirs(os.path.join(DATA_PATH_1, action, str(sequence)))
            except:
                pass


print('cration of folders done')
# %%
# Collecte Keypoints Value for trainig 

def collecte_keypoints_values():

    print('Openning the camera')
    cap = cv.VideoCapture(0)

    # Setting the mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as holistic:
        # Looping throw actions
        for action in actions:

            # Looping through videos
            for sequence in range(no_sequences):

                # Looping through frames
                for num_frame in range(sequence_lenght):

                    ret, frame = cap.read()
                    if ret == True:

                        # Make detections
                        image, results = mediapipe_detection(frame, holistic)

                        # Draw dtyled landmarkd
                        draw_styled_landmarks(image, results)

                        # Show information to screen 
                        if num_frame == 0:
                            cv.putText(
                                image, 'Starting Collection', (120,200),
                                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv.LINE_AA
                            )
                            cv.putText(
                                image, 'Collectiong frames for {} ## Video nuber {}'.format(action, sequence), (15, 12),
                                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA
                            )

                            # Show to screen 
                            # cv.imshow('collection of data', image)
                            cv.waitKey(2000)
                        
                        else:
                            cv.putText(image, 'Colecting frames for {} ## Video number {}'.format(action, sequence), (15, 12),
                                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0 ,255), 1, cv.LINE_AA)
                            
                            # Show to sceen
                        cv.imshow('Collecton of Data', image)

                        # Exporting Keypoints DATA into Folders
                        keypoints = extract_keypoints(results)
                        npy_path = os.path.join(DATA_PATH_1, action, str(sequence), str(num_frame))
                        np.save(npy_path, keypoints)

                    # Break 
                    if cv.waitKey(10) & 0xFF == ord('q'):
                        break
    cap.release()
    cv.destroyAllWindows()


# Preprocess Data and Create Labels and Features 
#%%
def process_data():
    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):

            window = []

            for num_frame in range(sequence_lenght):
                res = np.load(os.path.join(DATA_PATH_1, action, str(sequence), "{}.npy".format(num_frame)))
                window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

    print("shape of sequences array", np.array(sequences).shape)

    print("shape of labels array", np.array(labels).shape)
    return sequences, labels
#%%

def split_data(sequences, labels):
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    print("shap of X_train", np.shape(X_train))
    print("shap of X_test", np.shape(X_test))
    print("shap of y_train", np.shape(y_train))
    print("shap of y_test", np.shape(y_test))

    return X_train, X_test, y_train, y_test


#%%

# Build and Train the LSTM model 

def optimizer(params):

    return keras.optimizers.Adam(learning_rate=params)  # params=0.01

# Creating log directory and setting tensorboard for monitoring 

def monitor_training():
    log_dir = os.path.join('models/ModelAlex')
    tb_callback = TensorBoard(log_dir=log_dir)
    return log_dir, tb_callback

def buil_model():

# Setup & Build N.N Architecture

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    return model    


def train_model(model, params, X_train, y_train):

    optim = optimizer(params)

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    log_dir, tb_callback = monitor_training()


    model.fit(X_train, y_train, epochs=1800, callbacks=[tb_callback])

    model.summary()

    model.save(model_pth)


def test_model():
    # Test in real time
    model = keras.models.load_model('models/ModelTest.h5')

        # 1. New detection variable
    sequence = []
    sentence = []
    threshold = 0.5

    cap = cv.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            ret, frame = cap.read()
            # if ret == True:
            image , results = mediapipe_detection(frame, holistic)

            draw_styled_landmarks(image, results)

            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.insert(0, keypoints)
            sequence = sequence[:30]
            # sequence.append(keypoints)
            # sequence = sequence[-20:]

            if len(sequence) == 30:
                # sequence = [x + [0] * (len(sequence[0]) - len(x)) for x in sequence]
                # sequence_np = np.array([np.array(x) for x in sequence])
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])

            # 3. Visualisation logicq
        
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

#### main
#%%
if __name__ == "__main__":

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

    # test_model()

    #################### End testing Weights
# %%