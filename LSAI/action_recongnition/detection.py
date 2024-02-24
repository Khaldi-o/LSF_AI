# Importing and Installing needed Dependencied and Libriries
#%%
import cv2 as cv 
import numpy as np
import os
from matplotlib import pyplot as plt 
import time
import mediapipe as mp
# from params import *

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

from params import *
# Extracting Keypoints using MP Holistic
#%%
mp_holistic = mp.solutions.holistic     # Holistic Model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# the mediapipe_detection is the function to do detections
def mediapipe_detection(image, model):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)    # Color Conversion BGR to RGB
    image.flags.writeable = False                   # Image is no longer writeable
    results = model.process(image)                  # Make prediction ==> cette ligne detecte avec mediapipe / la variable image c'est une frame de opencv
    image.flags.writeable = True                    #Image is now writeable
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)    # Color conversion RGB to BGR
    return image, results # cette fonction retourne l'image et le résultats de la détection


# the draw_landmarks if the fct 
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)      # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)          # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)     # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)    # Draw Right hand connections
    

def draw_styled_landmarks(image, results):
    #Draw face connections
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=1),#la couleur du point
        mp_drawing.DrawingSpec(color=(80,256,121), thickness=2, circle_radius=1)# la couleur de la ligne de connection
    )


    # Draw pose connections
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=3),
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

##################### Applicate it with the camera ###############
#%%
cap = cv.VideoCapture(0) # take cammera Captures

# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic :
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)

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


# Extracting key point values
#%%
# print(results.pose_landmarks)
pose = []
for res in results.pose_landmarks.landmark:
    test = np.array([res.x, res.y, res.z, res.visibility])
    pose.append(test)

print('test est ', test)

pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

result_test = extract_keypoints(results)

print('result test are ', result_test)

np.save('5video_test', result_test)

np.load('5video_test.npy')

# print('len de face landmarks est ', len(results.face_landmarks.landmark)*3)

# Setup Folders for Collection
#%%
# DATA_PATH_1 = os.path.join('input/data_path_10video_assis')

# Actions that we try to detect

# actions = np.array(['Bonjour', 'Au revoir', 'Je', 'Tu', 'Etre', 'Avoir', 'Merci'])
# actions = np.array(['Bonjour', 'Au revoir', 'Je', 'etre', 'avoir'])


# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 20 frames in length
sequence_length = 30

for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH_2, action, str(sequence)))
        except:
            pass

#%%
# Collect Keypoint Values for Training and Testing

cap = cv.VideoCapture(0)

# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # NEW LOOP
    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(no_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
#                 print(results)

                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # NEW Apply wait logic
                if frame_num == 0: 
                    cv.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv.LINE_AA)
                    cv.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
                    # Show to screen
                    cv.imshow('OpenCV Feed', image)
                    cv.waitKey(2000)
                else: 
                    cv.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
                    # Show to screen
                    cv.imshow('OpenCV Feed', image)
                
                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH_2, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    cap.release()
    cv.destroyAllWindows()

cap.release()
cv.destroyAllWindows()

#%%
# Preprocess Data and Create Labels and Features 

label_map = {label:num for num, label in enumerate(actions)}
print(label_map)
print('ares est ', res)


sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH_1, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# print(np.array(sequences).shape)

# print(np.array(labels).shape)

X = np.array(sequences)
# print(X.shape)

y = to_categorical(labels).astype(int)
# print('la valeur de y est : ', y)

# print(' labels est :', labels)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# #%%
# # Build and Train LSTM N.N

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# actions[np.argmax(res)]

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])

model.summary()

# #%%
# # Make Predictions

# res = model.predict(X_test)

# # print(actions[np.argmax(res[4])])

# # print(actions[np.argmax(y_test[4])])

# # Save Weights

model.save('models/30_wave.h5')

# # del model

# # model.load_weights('models/Prez.etna2')

# #%%
# ############################## Evaluation using Confusion Matrix and Accuracy

# model.load_weights('models/10video_assis.h5')

# yhat = model.predict(X_test) # remplacer par x_train pour voir aussi

# ytrue = model.argmax(y_test, axis=1).tolist()
# yhat = np.argmax(yhat, axis=1).tolist()

# yhat #  [1, 1, 0, 0, 0]#
# ytrue

# multilabel_confusion_matrix(ytrue, yhat)
# #output[301]: array([[[2, 0],
# #                     [0, 3]],
# # 
# #                    [[3, 0],
# #                      [0,2]]], dtype=it64)
# #
# # multilabel_confusion_matrix??

# accuracy_score(ytrue, yhat) #   1.0 ==> 100%

# ############################## Test Model in real time 
# #%%

# sequence = []
# sentence = []
# threshold = 0.4

# cap = cv.VideoCapture(0) # take cammera Captures

# # Set mediapipe model
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic :
#     while cap.isOpened():

#         # Read feed
#         ret, frame = cap.read()

#         # Make detections
#         image, results = mediapipe_detection(frame, holistic)
#         print(results)

#         # Draw Landmarks
#         draw_styled_landmarks(image, results)

#         #   Prediction Logic
#         keypoints = extract_keypoints(results)
#         sequence.append(keypoints)
#         # sequence = [ :25]

#         if len(sequence) == 25:
#             res = model.predict(np.expand_dims(sequence, axis=0))[0]
#             print(res)


#         # Show to screen
#         cv.imshow('OpenCV Feed', image)

#         # Break gracefully 
#         if cv.waitKey(10) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv.destroyAllWindows()

# draw_landmarks(frame, results)
# plt.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))