import mediapipe as mp
import os
import numpy as np
import cv2 as cv

# Set up Mediapipe Holistic
mp_holistic = mp.solutions.holistic     # Holistic Model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Exctacted Data path

DATA_PATH_1 = os.path.join('models/ModelAlex')
DATA_PATH_2 = os.path.join('models/ModelAlex')
# Dictionary of Actions that we try to predict 

actions = np.array(['Bonjour', 'Au revoir', 'Je', 'etre', 'avoir'])


# VideoCapture variables
#==> Nuber of captured video 
no_sequences = 30

#==> number of frames per video 
sequence_lenght = 30


# Open the Camer
cap = cv.VideoCapture(0)


# Determine the mp holistic model params 

hol = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Label map
label_map = {label:num for num, label in enumerate(actions)}

# Folders of medels weight
# model_pth = os.path.join('models_weights')
model_pth = 'models/ModelTest.h5'
#Number of trainning epochs :
num_epochs = 200

test_size = 0.05