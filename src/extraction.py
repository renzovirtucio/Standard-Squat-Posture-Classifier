import glob
import os
import pandas as pd
from IPython.display import display
import cv2
import mediapipe as mp
import numpy as np
import math
# import custom_drawing_utils as mp_drawing
import utils
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
calculate_angle = utils.calculate_angle

CUSTOM_POSE_CONNECTIONS = frozenset([(11, 12), (11, 23), (12, 24), (23, 24), (23, 25),
                              (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                              (29, 31), (30, 32), (27, 31), (28, 32)])

## Get filepaths of videos according to class
acc_filepaths = [os.path.normpath(i) for i in glob.glob('D:\Documents\CS 198\Data Collection\Dataset\Segmented Videos\**/acc_0?.mp4', 
                   recursive = True)]
ak_filepaths = [os.path.normpath(i) for i in glob.glob('D:\Documents\CS 198\Data Collection\Dataset\Segmented Videos\**/ak_0?.mp4', 
                   recursive = True)]
bo_filepaths = [os.path.normpath(i) for i in glob.glob('D:\Documents\CS 198\Data Collection\Dataset\Segmented Videos\**/bo_0?.mp4', 
                   recursive = True)]
hs_filepaths = [os.path.normpath(i) for i in glob.glob('D:\Documents\CS 198\Data Collection\Dataset\Segmented Videos\**/hs_0?.mp4', 
                   recursive = True)]
kvg_filepaths = [os.path.normpath(i) for i in glob.glob('D:\Documents\CS 198\Data Collection\Dataset\Segmented Videos\**/kvg_0?.mp4', 
                   recursive = True)]
kvr_filepaths = [os.path.normpath(i) for i in glob.glob('D:\Documents\CS 198\Data Collection\Dataset\Segmented Videos\**/kvr_0?.mp4', 
                   recursive = True)]
compiled_filepaths = [acc_filepaths, ak_filepaths, bo_filepaths, hs_filepaths, kvg_filepaths, kvr_filepaths]

## Extract body landmark data per video
raw_extracted_data = []
for i in range(len(compiled_filepaths)):
  print(str(i) + " out of " + str(len(compiled_filepaths)-1))
  squat_class = []

  for filepath in compiled_filepaths[i]:
    print("Extracting data from " + filepath)
    
    repetition = []
    cap = cv2.VideoCapture(filepath)
    frame_num = 0
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
      while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
          frame50 = utils.rescale_frame(frame, 50)

          # Recolor image to RGB
          image = cv2.cvtColor(frame50, cv2.COLOR_BGR2RGB)
          image.flags.writeable = False

          # Make detection
          results = pose.process(image)

          # Recolor back to BGR
          image.flags.writeable = True
          # Render detections
          mp_drawing.draw_landmarks(image, results.pose_landmarks, CUSTOM_POSE_CONNECTIONS)
          image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

          try:
            # Extract landmarks
            landmarks = results.pose_landmarks.landmark
            landmark_data = []
            for j in range(33):
              landmark_data += [j, landmarks[j].x, landmarks[j].y, landmarks[j].visibility]
            
            repetition.append(landmark_data)

          except:
            pass
          
        else:
          break
    
    squat_class.append(repetition)
    cap.release()
    cv2.destroyAllWindows()

  raw_extracted_data.append(squat_class)

header = []
for i in range(33):
  header += ["body_" + str(i), "x_" + str(i), "y_" + str(i), "visibility_" + str(i)]

## Display sample dataframe
df = pd.DataFrame(raw_extracted_data[0][1], columns=header)
display(df)

## Save extracted data to .csv files
labels = ["acc", "ak", "bo", "hs", "kvg", "kvr"]
for i in range(len(labels)):
  for j in range(len(raw_extracted_data[i])):
    df = pd.DataFrame(raw_extracted_data[i][j], columns=header)
    df['target'] = labels[i]
    file_dest = "D:/Documents/CS 198/Data Collection/Dataset/Extracted Data/" + labels[i] + "/" + labels[i] + "_" + str(j) + ".csv"
    df.to_csv(path_or_buf=file_dest)