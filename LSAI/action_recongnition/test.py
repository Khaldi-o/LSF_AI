import cv2
import numpy as np
import os
from matplotlib import pyplot as plt 
import time
import mediapipe as mp
from params import *

import keras

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

from detection import mediapipe_detection, draw_styled_landmarks, extract_keypoints
from detection import draw_landmarks, actions

label_map = {label:num for num, label in enumerate(actions)}
print(label_map)
# print('ares est ', res)


sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_lenght):
            res = np.load(os.path.join(DATA_PATH_2, action, str(sequence), "{}.npy".format(frame_num)))
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
