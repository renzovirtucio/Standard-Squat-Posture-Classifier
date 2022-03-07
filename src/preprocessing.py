import pandas as pd
import glob
import os
import numpy as np
from IPython.display import display

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
import matplotlib.pyplot as plt

import cv2
import mediapipe as mp
import numpy as np
# import custom_drawing_utils as mp_drawing
import utils
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
calculate_angle = utils.calculate_angle

CUSTOM_POSE_CONNECTIONS = frozenset([(11, 12), (11, 23), (12, 24), (23, 24), (23, 25),
                              (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                              (29, 31), (30, 32), (27, 31), (28, 32)])

def load_data_into_df(filepaths):
  dataframes = []
  for filepath in filepaths:
    dataframes.append(pd.read_csv(filepath))

  return pd.concat(dataframes, ignore_index=True)

def modify_df(df):
  df = df.rename(columns={'Unnamed: 0': 'frame_number'})
  df = df.drop(['frame_number'], axis=1)

  for i in range(33):
      df = df.drop(['body_'+str(i)], axis=1)

  return df